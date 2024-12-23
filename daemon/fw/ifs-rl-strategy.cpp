/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014-2021,  Regents of the University of California,
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

#include <exception>
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
                            "IFSRLStrategy does not support version " + to_string(*parsed.version)));
                }
                this->setInstanceName(makeInstanceName(name, getStrategyName()));

                NFD_LOG_DEBUG("probing-interval=" << m_probing.getProbingInterval()
                << " n-silent-timeouts=" << m_nMaxSilentTimeouts);

                namespace python = boost::python;
                
                // Allow Python to load modules from the current directory.
                setenv("PYTHONPATH", "/home/vagrant/mini-ndn/ndn-src/NFD/daemon/fw/IFS-RL/", 1);

                // Initialize Python.
                Py_Initialize();
                
                python::object os = python::import("os");
                python::object sys = python::import("sys");
                
                python::object sys_path(sys.attr("path"));
                python::object sys_path_append(sys_path.attr("append"));
                sys_path_append("/home/vagrant/mini-ndn/ndn-src/NFD/daemon/fw/IFS-RL/");
                
                PyRun_SimpleString("import sys");
                PyRun_SimpleString("sys.path.append(\"/home/vagrant/mini-ndn/ndn-src/NFD/daemon/fw/IFS-RL/\")");
                
                sys_path(sys.attr("path"));
                python::object formatted_path = python::str(', ').join(sys_path);
                std::string str_path = python::extract<std::string>(formatted_path);
                NDN_LOG_INFO("Python error: " << str_path);
                
                try {
                    // Build the name object
                    PyObject *file_name = PyString_FromString((char*)"model_main");

                    // python::object my_python_class_module = python::import("from model_main import ModelMain");
                    PyObject *my_python_class_module = PyImport_Import(file_name);

                    if (my_python_class_module == nullptr) {
                        PyErr_Print();
                        NDN_LOG_INFO("Python error: No module !");
                    }

                    // dict is a borrowed reference.
                    PyObject *my_python_class_dict = PyModule_GetDict(my_python_class_module);
                    if (my_python_class_dict == nullptr) {
                        PyErr_Print();
                        NDN_LOG_INFO("Python error: Fails to get the dictionary.");
                    }

                    // Builds the name of a callable class
                    // python::object model_main_class = my_python_class_module.attr("ModelMain")();
                    PyObject *model_main_class = PyDict_GetItemString(dict, "ModelMain");
                    if (python_class == nullptr) {
                        PyErr_Print();
                        NDN_LOG_INFO("Fails to get the Python class.");
                    }

                    // Creates an instance of the class
                    if (PyCallable_Check(model_main_class)) {
                        PyObject *object = PyObject_CallObject(model_main_class, nullptr);
                    } else {
                        NDN_LOG_INFO("Cannot instantiate the Python class");
                    }
                }
                catch (python::error_already_set& e) {
                    // https://stackoverflow.com/a/1418703/2049763
                    NDN_LOG_INFO("Python error: Inside catch 1");
                    PyErr_Print();
                    
                    PyObject *error_type = NULL,*error_value = NULL,*error_traceback = NULL;
                    python::object formatted_list, formatted;
                    PyErr_Fetch(&error_type,&error_value,&error_traceback);
                    PyErr_NormalizeException(&error_type,&error_value,&error_traceback);
                    
                    python::handle<> hexc(python::allow_null(error_type)),hval(python::allow_null(error_value)),htb(python::allow_null(error_traceback)); 
                    python::object traceback = python::import("traceback");
                    python::object format_exception(traceback.attr("format_exception"));
                    NDN_LOG_INFO("Python error: Inside catch 85");
                    formatted_list = format_exception(hexc,hval,htb);
                    NDN_LOG_INFO("Python error: Inside catch 87");
                    
                    formatted = python::str('\n').join(formatted_list);
                    NDN_LOG_INFO("Python error: Inside catch 90");
                    std::string errorSummary_ = python::extract<std::string>(formatted);
                    NDN_LOG_INFO("Python error: " << errorSummary_);
                    
                    python::object format_exc(traceback.attr("format_exc"));
                    formatted = format_exc();
                    errorSummary_ = python::extract<std::string>(formatted);
                    NDN_LOG_INFO("Python error: " << errorSummary_);                    
                }
//                try {
//                    python::object model_main_class = my_python_class_module.attr("ModelMain")();
//                }
//                catch (python::error_already_set& e) {
//                    NDN_LOG_INFO("Python error: Inside catch 2");
//                    PyErr_Print();
//
//                    PyObject *error_type = NULL,*error_value = NULL,*error_traceback = NULL;
//                    python::object formatted_list, formatted;
//                    PyErr_Fetch(&error_type,&error_value,&error_traceback);
//                    PyErr_NormalizeException(&error_type,&error_value,&error_traceback);
//
//                    python::handle<> hexc(python::allow_null(error_type)),hval(python::allow_null(error_value)),htb(python::allow_null(error_traceback));
//                    python::object traceback = python::import("traceback");
//                    python::object format_exception(traceback.attr("format_exception"));
//                    NDN_LOG_INFO("Python error: Inside catch 85");
//                    formatted_list = format_exception(hexc,hval,htb);
//                    NDN_LOG_INFO("Python error: Inside catch 87");
//
//                    formatted = python::str('\n').join(formatted_list);
//                    NDN_LOG_INFO("Python error: Inside catch 90");
//                    std::string errorSummary_ = python::extract<std::string>(formatted);
//                    NDN_LOG_INFO("Python error: " << errorSummary_);
//
//                    python::object format_exc(traceback.attr("format_exc"));
//                    formatted = format_exc();
//                    errorSummary_ = python::extract<std::string>(formatted);
//                    NDN_LOG_INFO("Python error: " << errorSummary_);
//                }
            }

            const Name &
            IFSRLStrategy::getStrategyName() {
                static const auto strategyName = Name("/localhost/nfd/strategy/ifs-rl").appendVersion(4);
                return strategyName;
            }

            static uint64_t
            getParamValue(const std::string &param, const std::string &value) {
                try {
                    if (!value.empty() && value[0] == '-')
                        NDN_THROW(boost::bad_lexical_cast());

                    return boost::lexical_cast<uint64_t>(value);
                }
                catch (const boost::bad_lexical_cast &) {
                    NDN_THROW(std::invalid_argument("Value of " + param + " must be a non-negative integer"));
                }
            }

            void
            IFSRLStrategy::processParams(const PartialName &parsed) {
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
                        m_nMaxSilentTimeouts = getParamValue(f, s);
                    } else {
                        NDN_THROW(std::invalid_argument("Parameter should be probing-interval or n-silent-timeouts"));
                    }
                }
            }

            /*
            * afterReceiveInterest
            *
            * @see page 40 of documents for details
            * @param interest Interest packet
            * @param ingress its incoming face
            * @param pitEntry the PIT entry
            * */
            void
            IFSRLStrategy::afterReceiveInterest(const Interest &interest, const FaceEndpoint &ingress,
                                                const shared_ptr <pit::Entry> &pitEntry) {
                const auto &fibEntry = this->lookupFib(*pitEntry);

                // Check if the interest is new and, if so, skip the retx suppression check
                if (!hasPendingOutRecords(*pitEntry)) {
                    auto *faceToUse = getBestFaceForForwarding(interest, ingress.face, fibEntry, pitEntry);
                    if (faceToUse == nullptr) {
                        NFD_LOG_DEBUG(interest << " new-interest from=" << ingress << " no-nexthop");
                        sendNoRouteNack(ingress.face, pitEntry);
                    } else {
                        NFD_LOG_DEBUG(
                                interest << " new-interest from=" << ingress << " forward-to=" << faceToUse->getId());
                        forwardInterest(interest, *faceToUse, fibEntry, pitEntry);
                        sendProbe(interest, ingress, *faceToUse, fibEntry, pitEntry);
                    }
                    return;
                }

                auto *faceToUse = getBestFaceForForwarding(interest, ingress.face, fibEntry, pitEntry, false);
                if (faceToUse != nullptr) {
                    auto suppressResult = m_retxSuppression.decidePerUpstream(*pitEntry, *faceToUse);
                    if (suppressResult == RetxSuppressionResult::SUPPRESS) {
                        // Cannot be sent on this face, interest was received within the suppression window
                        NFD_LOG_DEBUG(interest << " retx-interest from=" << ingress
                        << " forward-to=" << faceToUse->getId() << " suppressed");
                    } else {
                        // The retx arrived after the suppression period: forward it but don't probe, because
                        // probing was done earlier for this interest when it was newly received
                        NFD_LOG_DEBUG(
                                interest << " retx-interest from=" << ingress << " forward-to=" << faceToUse->getId());
                        auto *outRecord = forwardInterest(interest, *faceToUse, fibEntry, pitEntry);
                        if (outRecord && suppressResult == RetxSuppressionResult::FORWARD) {
                            m_retxSuppression.incrementIntervalForOutRecord(*outRecord);
                        }
                    }
                    return;
                }

                // If all eligible faces have been used (i.e., they all have a pending out-record),
                // choose the nexthop with the earliest out-record
                const auto &nexthops = fibEntry.getNextHops();
                auto it = findEligibleNextHopWithEarliestOutRecord(ingress.face, interest, nexthops, pitEntry);
                if (it == nexthops.end()) {
                    NFD_LOG_DEBUG(interest << " retx-interest from=" << ingress << " no eligible nexthop");
                    return;
                }
                auto &outFace = it->getFace();
                auto suppressResult = m_retxSuppression.decidePerUpstream(*pitEntry, outFace);
                if (suppressResult == RetxSuppressionResult::SUPPRESS) {
                    NFD_LOG_DEBUG(interest << " retx-interest from=" << ingress
                    << " retry-to=" << outFace.getId() << " suppressed");
                } else {
                    NFD_LOG_DEBUG(interest << " retx-interest from=" << ingress << " retry-to=" << outFace.getId());
                    // sendInterest() is used here instead of forwardInterest() because the measurements info
                    // were already attached to this face in the previous forwarding
                    auto *outRecord = sendInterest(interest, outFace, pitEntry);
                    if (outRecord && suppressResult == RetxSuppressionResult::FORWARD) {
                        m_retxSuppression.incrementIntervalForOutRecord(*outRecord);
                    }
                }
            }

            void
            IFSRLStrategy::beforeSatisfyInterest(const Data &data, const FaceEndpoint &ingress,
                                                 const shared_ptr <pit::Entry> &pitEntry) {
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
                    const auto &name_prefix = pitEntry->getName();
                    namespace python = boost::python;
                    try {
                        python::object result = model_main_class.attr("send_face_forwarding_metrics")(name_prefix,
                                ingress.face,
                                time::steady_clock::now() -
                                outRecord->getLastRenewed());
                    }
                    catch (const python::error_already_set &) {
                        PyErr_Print();
                    }
                }

                // Extend lifetime for measurements associated with Face
                namespaceInfo->extendFaceInfoLifetime(*faceInfo, ingress.face.getId());
                // Extend PIT entry timer to allow slower probes to arrive
                this->setExpiryTimer(pitEntry, 50_ms);
                faceInfo->cancelTimeout(data.getName());
            }

            void
            IFSRLStrategy::afterReceiveNack(const lp::Nack &nack, const FaceEndpoint &ingress,
                                            const shared_ptr <pit::Entry> &pitEntry) {
                NFD_LOG_DEBUG(nack.getInterest() << " nack from=" << ingress << " reason=" << nack.getReason());
                onTimeoutOrNack(pitEntry->getName(), ingress.face.getId(), true);
            }

            pit::OutRecord *
            IFSRLStrategy::forwardInterest(const Interest &interest, Face &outFace,
                                           const fib::Entry &fibEntry,
                                           const shared_ptr <pit::Entry> &pitEntry) {
                const auto &interestName = interest.getName();
                auto faceId = outFace.getId();

                auto *outRecord = sendInterest(interest, outFace, pitEntry);

                FaceInfo &faceInfo = m_measurements.getOrCreateFaceInfo(fibEntry, interestName, faceId);

                // Refresh measurements since Face is being used for forwarding
                NamespaceInfo &namespaceInfo = m_measurements.getOrCreateNamespaceInfo(fibEntry, interestName);
                namespaceInfo.extendFaceInfoLifetime(faceInfo, faceId);

                if (!faceInfo.isTimeoutScheduled()) {
                    auto timeout = faceInfo.scheduleTimeout(interestName,
                                                            [this, name = interestName, faceId] {
                        onTimeoutOrNack(name, faceId, false);
                    });
                    NFD_LOG_TRACE("Scheduled timeout for " << fibEntry.getPrefix() << " to=" << faceId
                    << " in "
                    << time::duration_cast<time::milliseconds>(timeout));
                }

                return outRecord;
            }

            void
            IFSRLStrategy::sendProbe(const Interest &interest, const FaceEndpoint &ingress, const Face &faceToUse,
                                     const fib::Entry &fibEntry, const shared_ptr <pit::Entry> &pitEntry) {
                if (!m_probing.isProbingNeeded(fibEntry, interest.getName()))
                    return;

                Face *faceToProbe = m_probing.getFaceToProbe(ingress.face, interest, fibEntry, faceToUse);
                if (faceToProbe == nullptr)
                    return;

                Interest probeInterest(interest);
                probeInterest.refreshNonce();
                NFD_LOG_TRACE("Sending probe " << probeInterest << " to=" << faceToProbe->getId());
                forwardInterest(probeInterest, *faceToProbe, fibEntry, pitEntry);

                m_probing.afterForwardingProbe(fibEntry, interest.getName());
            }

            struct FaceStats {
                Face *face;
                time::nanoseconds rtt;
                time::nanoseconds srtt;
                uint64_t cost;
            };

            struct FaceStatsCompare {
                bool operator()(const FaceStats &lhs, const FaceStats &rhs) const {
                    time::nanoseconds lhsValue = getValueForSorting(lhs);
                    time::nanoseconds rhsValue = getValueForSorting(rhs);

                    // Sort by RTT and then by cost
                    return std::tie(lhsValue, lhs.cost) < std::tie(rhsValue, rhs.cost);
                }

            private:
                static
                time::nanoseconds getValueForSorting(const FaceStats &stats) {
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

            Face *
            IFSRLStrategy::getBestFaceForForwarding(const Interest &interest, const Face &inFace,
                                                    const fib::Entry &fibEntry,
                                                    const shared_ptr <pit::Entry> &pitEntry,
                                                    bool isInterestNew) {
                std::set <FaceStats, FaceStatsCompare> rankedFaces;

                const auto &name_prefix = fibEntry.getPrefix();
                namespace python = boost::python;
                std::string prefix_face_status;
                FaceId best_prefix_face_id;
                try {
                    python::object result = model_main_class.attr("get_prefix_face_status")(name_prefix);
                    prefix_face_status = python::extract<std::string>(result);
                    if (prefix_face_status == "RESULT_READY") {
                        NDN_LOG_INFO("Result is ready");
                        python::object result = model_main_class.attr("get_prefix_face_result")(name_prefix);
                        best_prefix_face_id = python::extract<FaceId>(result);
                        NDN_LOG_INFO("RR: got best face, face id " << best_prefix_face_id << " for the prefix: "
                        << name_prefix);
                    }
                } catch (const python::error_already_set &) {
                    PyErr_Print();
                    NDN_LOG_INFO("Result not ready or no information");
                    prefix_face_status = "NO_INFORMATION";
                }

                auto now = time::steady_clock::now();
                for (const auto &nh : fibEntry.getNextHops()) {
                    if (!isNextHopEligible(inFace, interest, nh, pitEntry, !isInterestNew, now)) {
                        continue;
                    }
                    if (prefix_face_status == "RESULT_READY") {
                        NDN_LOG_INFO("Result is ready, returning best face");
                        if (best_prefix_face_id == nh.getFace().getId()) {
                            NDN_LOG_INFO("RR: Returing face for the face-id: " << best_prefix_face_id
                            << " and for the prefix: "
                            << name_prefix);
                            return &nh.getFace();
                        }
                    }
                    const FaceInfo *info = m_measurements.getFaceInfo(fibEntry, interest.getName(),
                                                                      nh.getFace().getId());
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
                        NDN_LOG_INFO("Inside catch 2");
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
                        python::object result = model_main_class.attr("calculate_prefix_face_result")(name_prefix);
                        FaceId best_prefix_face_id = python::extract<FaceId>(result);
                        NDN_LOG_INFO(
                                "calculate_prefix_face_result: Returing face for the face-id: " << best_prefix_face_id
                                << " prefix: "
                                << name_prefix);
                        // check if best_prefix_face_id is null or not
                        for (const auto &nh : fibEntry.getNextHops()) {
                            if (best_prefix_face_id == nh.getFace().getId())
                                return &nh.getFace();
                        }
                    } catch (const python::error_already_set &) {
                        PyErr_Print();
                    }
                }
                auto it = rankedFaces.begin();
                return it != rankedFaces.end() ? it->face : nullptr;
            }
            // ######################################################################################

            void
            IFSRLStrategy::onTimeoutOrNack(const Name &interestName, FaceId faceId, bool isNack) {
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

                if (nTimeouts <= m_nMaxSilentTimeouts && !isNack) {
                    NFD_LOG_TRACE(interestName << " face=" << faceId << " timeout-count=" << nTimeouts << " ignoring");
                    // Extend lifetime for measurements associated with Face
                    namespaceInfo->extendFaceInfoLifetime(faceInfo, faceId);
                    faceInfo.cancelTimeout(interestName);
                } else {
                    NFD_LOG_TRACE(interestName << " face=" << faceId << " timeout-count=" << nTimeouts);
                    faceInfo.recordTimeout(interestName);

                    namespace python = boost::python;
                    try {
                        python::object result = model_main_class.attr("send_face_forwarding_metrics")(interestName,
                                faceId, 100);
                    } catch (const python::error_already_set &) {
                        PyErr_Print();
                    }
                }
            }

            void
            IFSRLStrategy::sendNoRouteNack(Face &face, const shared_ptr <pit::Entry> &pitEntry) {
                lp::NackHeader nackHeader;
                nackHeader.setReason(lp::NackReason::NO_ROUTE);
                this->sendNack(nackHeader, face, pitEntry);
                this->rejectPendingInterest(pitEntry);
            }
        } // namespace ifs
    } // namespace fw
} // namespace nfd
