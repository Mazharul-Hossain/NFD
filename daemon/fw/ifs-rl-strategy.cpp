/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014-2019,  Regents of the University of California,
 *                           Arizona Board of Regents,
 *                           Colorado State University,
 *                           University Pierre & Marie Curie, Sorbonne University,
 *                           Washington University in St. Louis,
 *                           Beijing Institute of Technology,
 *                           The University of Memphis.
 *
 * This file is part of NFD (Named Data Networking Forwarding Daemon).
 * See AUTHORS.md for complete list of NFD authors and contributors.
 *
 * NFD is free software: you can redistribute it and/or modify it under the terms
 * of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * NFD is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * NFD, e.g., in COPYING.md file.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "ifs-rl-strategy.hpp"
#include "algorithm.hpp"
#include "common/global.hpp"
#include "common/logger.hpp"

namespace nfd {
    namespace fw {
        namespace asf {

            NFD_LOG_INIT(IFSRLStrategy);
            NFD_REGISTER_STRATEGY(IFSRLStrategy);

            const time::milliseconds IFSRLStrategy::RETX_SUPPRESSION_INITIAL(10);
            const time::milliseconds IFSRLStrategy::RETX_SUPPRESSION_MAX(250);

            /*
             * IFSRLStrategy class initializer
             *
             * @param Forwarder host identification
             * @param Name Strategy Name
             * */
            IFSRLStrategy::IFSRLStrategy(Forwarder &forwarder, const Name &name)
                    : Strategy(forwarder), m_measurements(getMeasurements()), m_probing(m_measurements),
                      m_retxSuppression(RETX_SUPPRESSION_INITIAL, RetxSuppressionExponential::DEFAULT_MULTIPLIER,
                                        RETX_SUPPRESSION_MAX) {

                ParsedInstanceName parsed = parseInstanceName(name);
                if (!parsed.parameters.empty()) {
                    processParams(parsed.parameters);
                }

                if (parsed.version && *parsed.version != getStrategyName()[-1].toVersion()) {
                    NDN_THROW(std::invalid_argument(
                            "AsfStrategy does not support version " + to_string(*parsed.version)));
                }
                this->setInstanceName(makeInstanceName(name, getStrategyName()));

                // Allow Python to load modules from the current directory.
                setenv("PYTHONPATH", "./IFS-RL", 1);
                // Initialize Python.
                Py_Initialize();

                NFD_LOG_DEBUG("probing-interval=" << m_probing.getProbingInterval()
                                                  << " n-silent-timeouts=" << m_maxSilentTimeouts);

                namespace python = boost::python;
                try {
                    // >>> import MyPythonClass
                    my_python_class_module = python::import("model_main");

                    // >>> dog = MyPythonClass.Dog()
                    model_main_class = my_python_class_module.attr("ModelMain")();

                    // >>> dog.bark("woof");
                    // dog.attr("bark")("woof");
                }
                catch (const python::error_already_set &) {
                    PyErr_Print();
                }
            }

            const Name &IFSRLStrategy::getStrategyName() {
                static Name strategyName("/localhost/nfd/strategy/ifs-rl/%FD%03");
                return strategyName;
            }

            static uint64_t getParamValue(const std::string &param, const std::string &value) {
                try {
                    if (!value.empty() && value[0] == '-')
                        NDN_THROW(boost::bad_lexical_cast());

                    return boost::lexical_cast<uint64_t>(value);
                }
                catch (const boost::bad_lexical_cast &) {
                    NDN_THROW(std::invalid_argument("Value of " + param + " must be a non-negative integer"));
                }
            }

            void IFSRLStrategy::processParams(const PartialName &parsed) {
                for (const auto &component : parsed) {
                    std::string parsedStr(reinterpret_cast<const char *>(component.value()), component.value_size());
                    auto n = parsedStr.find("~");
                    if (n == std::string::npos) {
                        NDN_THROW(std::invalid_argument("Format is <parameter>~<value>"));
                    }

                    auto f = parsedStr.substr(0, n);
                    auto s = parsedStr.substr(n + 1);
                    if (f == "probing-interval") {
                        m_probing.setProbingInterval(getParamValue(f, s));
                    } else if (f == "n-silent-timeouts") {
                        m_maxSilentTimeouts = getParamValue(f, s);
                    } else {
                        NDN_THROW(std::invalid_argument("Parameter should be probing-interval or n-silent-timeouts"));
                    }
                }
            }

            /*
             * afterReceiveInterest
             *
             * @see page 40 of documents for details
             *
             * @param ingress its incoming face
             * @param interest Interest packet
             * @param pitEntry the PIT entry
             * */
            void IFSRLStrategy::afterReceiveInterest(const FaceEndpoint &ingress, const Interest &interest,
                                                     const shared_ptr <pit::Entry> &pitEntry) {
                // Should the Interest be suppressed?
                auto suppressResult = m_retxSuppression.decidePerPitEntry(*pitEntry);
                if (suppressResult == RetxSuppressionResult::SUPPRESS) {
                    NFD_LOG_DEBUG(interest << " retx-interest from=" << ingress << " suppressed");
                    return;
                }

                const fib::Entry &fibEntry = this->lookupFib(*pitEntry);
                const fib::NextHopList &nexthops = fibEntry.getNextHops();

                if (suppressResult == RetxSuppressionResult::NEW) {
                    if (nexthops.size() == 0) {
                        NFD_LOG_DEBUG(interest << " new-interest from=" << ingress << " no-nexthop");
                        sendNoRouteNack(ingress, pitEntry);
                        return;
                    }

                    Face *faceToUse = getBestFaceForForwarding(interest, ingress.face, fibEntry, pitEntry);
                    if (faceToUse != nullptr) {
                        NFD_LOG_DEBUG(
                                interest << " new-interest from=" << ingress << " forward-to=" << faceToUse->getId());
                        forwardInterest(interest, *faceToUse, fibEntry, pitEntry);

                        // If necessary, send probe
                        sendProbe(interest, ingress, *faceToUse, fibEntry, pitEntry);
                    } else {
                        NFD_LOG_DEBUG(interest << " new-interest from=" << ingress << " no-nexthop");
                        sendNoRouteNack(ingress, pitEntry);
                    }
                    return;
                }

                Face *faceToUse = getBestFaceForForwarding(interest, ingress.face, fibEntry, pitEntry, false);
                // if unused face not found, select nexthop with earliest out record
                if (faceToUse != nullptr) {
                    NFD_LOG_DEBUG(
                            interest << " retx-interest from=" << ingress << " forward-to=" << faceToUse->getId());
                    forwardInterest(interest, *faceToUse, fibEntry, pitEntry);
                    // avoid probing in case of forwarding
                    return;
                }

                // find an eligible upstream that is used earliest
                auto it = nexthops.end();
                it = findEligibleNextHopWithEarliestOutRecord(ingress.face, interest, nexthops, pitEntry);
                if (it == nexthops.end()) {
                    NFD_LOG_DEBUG(interest << " retx-interest from=" << ingress << " no-nexthop");
                } else {
                    auto egress = FaceEndpoint(it->getFace(), 0);
                    NFD_LOG_DEBUG(interest << " retx-interest from=" << ingress << " retry-to=" << egress);
                    this->sendInterest(pitEntry, egress, interest);
                }
            }

            void IFSRLStrategy::beforeSatisfyInterest(const shared_ptr <pit::Entry> &pitEntry,
                                                      const FaceEndpoint &ingress, const Data &data) {

                NamespaceInfo *namespaceInfo = m_measurements.getNamespaceInfo(pitEntry->getName());

                if (namespaceInfo == nullptr) {
                    NFD_LOG_DEBUG(pitEntry->getName() << " data from=" << ingress << " no-measurements");
                    return;
                }

                // Record the RTT between the Interest out to Data in
                FaceInfo *faceInfo = namespaceInfo->getFaceInfo(ingress.face.getId());
                if (faceInfo == nullptr) {
                    NFD_LOG_DEBUG(pitEntry->getName() << " data from=" << ingress << " no-face-info");
                    return;
                }

                auto outRecord = pitEntry->getOutRecord(ingress.face);
                if (outRecord == pitEntry->out_end()) {
                    NFD_LOG_DEBUG(pitEntry->getName() << " data from=" << ingress << " no-out-record");
                } else {
                    faceInfo->recordRtt(time::steady_clock::now() - outRecord->getLastRenewed());
                    NFD_LOG_DEBUG(pitEntry->getName() << " data from=" << ingress
                                                      << " rtt=" << faceInfo->getLastRtt() << " srtt="
                                                      << faceInfo->getSrtt());
                    // needs change !!!
                    const &name_prefix = pitEntry->getName();
                    namespace python = boost::python;
                    try {
                        object result = model_main_class.attr("send_face_forwarding_metrics")(name_prefix, ingress.face,
                                                                                              time::steady_clock::now() -
                                                                                              outRecord->getLastRenewed());
                    }
                    catch (const python::error_already_set &) {
                        PyErr_Print();
                    }
                }

                // Extend lifetime for measurements associated with Face
                namespaceInfo->extendFaceInfoLifetime(*faceInfo, ingress.face.getId());

                faceInfo->cancelTimeout(data.getName());
            }

            void IFSRLStrategy::afterReceiveNack(const FaceEndpoint &ingress, const lp::Nack &nack,
                                                 const shared_ptr <pit::Entry> &pitEntry) {
                NFD_LOG_DEBUG(nack.getInterest() << " nack from=" << ingress << " reason=" << nack.getReason());
                onTimeout(pitEntry->getName(), ingress.face.getId());
            }

            void IFSRLStrategy::forwardInterest(const Interest &interest, Face &outFace, const fib::Entry &fibEntry,
                                                const shared_ptr <pit::Entry> &pitEntry, bool wantNewNonce) {
                auto egress = FaceEndpoint(outFace, 0);
                if (wantNewNonce) {
                    // Send probe: interest with new Nonce
                    Interest probeInterest(interest);
                    probeInterest.refreshNonce();
                    NFD_LOG_TRACE("Sending probe for " << probeInterest << " to=" << egress);
                    this->sendInterest(pitEntry, egress, probeInterest);
                } else {
                    this->sendInterest(pitEntry, egress, interest);
                }

                FaceInfo &faceInfo = m_measurements.getOrCreateFaceInfo(fibEntry, interest, egress.face.getId());

                // Refresh measurements since Face is being used for forwarding
                NamespaceInfo &namespaceInfo = m_measurements.getOrCreateNamespaceInfo(fibEntry, interest);
                namespaceInfo.extendFaceInfoLifetime(faceInfo, egress.face.getId());

                if (!faceInfo.isTimeoutScheduled()) {
                    auto timeout = faceInfo.scheduleTimeout(interest.getName(),
                                                            [this, name = interest.getName(), faceId = egress.face.getId()] {
                                                                onTimeout(name, faceId);
                                                            });
                    NFD_LOG_TRACE("Scheduled timeout for " << fibEntry.getPrefix() << " to=" << egress
                                                           << " in " << time::duration_cast<time::milliseconds>(timeout)
                                                           << " ms");
                }
            }

            void IFSRLStrategy::sendProbe(const Interest &interest, const FaceEndpoint &ingress, const Face &faceToUse,
                                          const fib::Entry &fibEntry, const shared_ptr <pit::Entry> &pitEntry) {
                if (!m_probing.isProbingNeeded(fibEntry, interest))
                    return;

                Face *faceToProbe = m_probing.getFaceToProbe(ingress.face, interest, fibEntry, faceToUse);
                if (faceToProbe == nullptr)
                    return;

                forwardInterest(interest, *faceToProbe, fibEntry, pitEntry, true);
                m_probing.afterForwardingProbe(fibEntry, interest);
            }

            struct FaceStats {
                Face *face;
                time::nanoseconds rtt;
                time::nanoseconds srtt;
                uint64_t cost;
            };

            struct FaceStatsCompare {
                bool
                operator()(const FaceStats &lhs, const FaceStats &rhs) const {
                    time::nanoseconds lhsValue = getValueForSorting(lhs);
                    time::nanoseconds rhsValue = getValueForSorting(rhs);

                    // Sort by RTT and then by cost
                    return std::tie(lhsValue, lhs.cost) < std::tie(rhsValue, rhs.cost);
                }

            private:
                static time::nanoseconds
                getValueForSorting(const FaceStats &stats) {
                    // These values allow faces with no measurements to be ranked better than timeouts
                    // srtt < RTT_NO_MEASUREMENT < RTT_TIMEOUT
                    if (stats.rtt == FaceInfo::RTT_TIMEOUT) {
                        return time::nanoseconds::max();
                    } else if (stats.rtt == FaceInfo::RTT_NO_MEASUREMENT) {
                        return time::nanoseconds::max() / 2;
                    } else {
                        return stats.srtt;
                    }
                }
            };

            /*
             * This method needs to be changed
             * */
            Face *IFSRLStrategy::getBestFaceForForwarding(const Interest &interest, const Face &inFace,
                                                          const fib::Entry &fibEntry,
                                                          const shared_ptr <pit::Entry> &pitEntry,
                                                          bool isInterestNew) {
                std::set <FaceStats, FaceStatsCompare> rankedFaces;

                const &name_prefix = fibEntry.getPrefix();
                namespace python = boost::python;
                try {
                    object result = model_main_class.attr("get_prefix_face_status")(name_prefix);
                    str prefix_face_status = extract<str>(result);
                    if (prefix_face_status == "RESULT_READY") {
                        object result = model_main_class.attr("get_prefix_face_result")(name_prefix);
                        Face best_prefix_face = extract<Face>(result);
                    }
                }
                catch (const python::error_already_set &) {
                    PyErr_Print();
                    str prefix_face_status = "NO_INFORMATION";
                }

                auto now = time::steady_clock::now();
                for (const auto &nh : fibEntry.getNextHops()) {
                    if (!isNextHopEligible(inFace, interest, nh, pitEntry, !isInterestNew, now)) {
                        continue;
                    }
                    if (prefix_face_status == "RESULT_READY") {
                        if (best_prefix_face == nh.getFace())
                            return nh.getFace()
                    }
                    FaceInfo *info = m_measurements.getFaceInfo(fibEntry, interest, nh.getFace().getId());
                    try {
                        if (info == nullptr) {
                            model_main_class.attr("face_info_for_ranking")(name_prefix, &nh.getFace(),
                                                                           FaceInfo::RTT_NO_MEASUREMENT,
                                                                           FaceInfo::RTT_NO_MEASUREMENT,
                                                                           nh.getCost(), "null");
                        } else {
                            model_main_class.attr("face_info_for_ranking")(name_prefix, &nh.getFace(),
                                                                           info->getLastRtt(),
                                                                           info->getSrtt(), nh.getCost(), "");
                        }
                    }
                    catch (const python::error_already_set &) {
                        PyErr_Print();
                    }
                    if (info == nullptr) {
                        rankedFaces.insert({&nh.getFace(), FaceInfo::RTT_NO_MEASUREMENT,
                                            FaceInfo::RTT_NO_MEASUREMENT, nh.getCost()});
                    } else {
                        rankedFaces.insert({&nh.getFace(), info->getLastRtt(), info->getSrtt(), nh.getCost()});
                    }
                }
                if (prefix_face_status == "READY_FOR_CALCULATION") {
                    try {
                        object result = model_main_class.attr("calculate_prefix_face_result")(name_prefix);
                        Face best_prefix_face = extract<Face>(result);
                        if (best_prefix_face != nullptr) {
                            return best_prefix_face
                        }
                    }
                    catch (const python::error_already_set &) {
                        PyErr_Print();
                    }
                }
                auto it = rankedFaces.begin();
                return it != rankedFaces.end() ? it->face : nullptr;
            }
            // ######################################################################################

            void IFSRLStrategy::onTimeout(const Name &interestName, FaceId faceId) {
                NamespaceInfo *namespaceInfo = m_measurements.getNamespaceInfo(interestName);
                if (namespaceInfo == nullptr) {
                    NFD_LOG_TRACE(interestName << " FibEntry has been removed since timeout scheduling");
                    return;
                }

                FaceInfo *fiPtr = namespaceInfo->getFaceInfo(faceId);
                if (fiPtr == nullptr) {
                    NFD_LOG_TRACE(
                            interestName << " FaceInfo id=" << faceId << " has been removed since timeout scheduling");
                    return;
                }

                auto &faceInfo = *fiPtr;
                size_t nTimeouts = faceInfo.getNSilentTimeouts() + 1;
                faceInfo.setNSilentTimeouts(nTimeouts);

                if (nTimeouts <= m_maxSilentTimeouts) {
                    NFD_LOG_TRACE(interestName << " face=" << faceId << " timeout-count=" << nTimeouts << " ignoring");
                    // Extend lifetime for measurements associated with Face
                    namespaceInfo->extendFaceInfoLifetime(faceInfo, faceId);
                    faceInfo.cancelTimeout(interestName);
                } else {
                    NFD_LOG_TRACE(interestName << " face=" << faceId << " timeout-count=" << nTimeouts);
                    faceInfo.recordTimeout(interestName);

                    // needs change !!!
                    const &name_prefix = interestName;
                    namespace python = boost::python;
                    try {
                        object result = model_main_class.attr("send_face_forwarding_metrics")(name_prefix, faceId, 100);
                    }
                    catch (const python::error_already_set &) {
                        PyErr_Print();
                    }
                }
            }

            void IFSRLStrategy::sendNoRouteNack(const FaceEndpoint &ingress, const shared_ptr <pit::Entry> &pitEntry) {
                lp::NackHeader nackHeader;
                nackHeader.setReason(lp::NackReason::NO_ROUTE);
                this->sendNack(pitEntry, ingress, nackHeader);
                this->rejectPendingInterest(pitEntry);
            }
        } // namespace asf
    } // namespace fw
} // namespace nfd
