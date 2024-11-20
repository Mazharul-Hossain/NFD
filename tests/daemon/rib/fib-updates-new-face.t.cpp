/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014-2024,  Regents of the University of California,
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

#include "rib/rib.hpp"

#include "tests/test-common.hpp"
#include "fib-updates-common.hpp"

namespace nfd::tests {

BOOST_AUTO_TEST_SUITE(Rib)
BOOST_FIXTURE_TEST_SUITE(TestFibUpdates, FibUpdatesFixture)
BOOST_AUTO_TEST_SUITE(NewFace)

BOOST_AUTO_TEST_CASE(Basic)
{
  // should generate 1 update
  insertRoute("/", 1, 0, 50, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);

  FibUpdater::FibUpdateList updates = getFibUpdates();
  BOOST_REQUIRE_EQUAL(updates.size(), 1);

  FibUpdater::FibUpdateList::const_iterator update = updates.begin();

  BOOST_CHECK_EQUAL(update->name,  "/");
  BOOST_CHECK_EQUAL(update->faceId, 1);
  BOOST_CHECK_EQUAL(update->cost,   50);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);

  // Clear any updates generated from previous insertions
  clearFibUpdates();

  // should generate 2 updates
  insertRoute("/a", 2, 0, 50, 0);

  updates = getSortedFibUpdates();
  BOOST_REQUIRE_EQUAL(updates.size(), 2);

  update = updates.begin();
  BOOST_CHECK_EQUAL(update->name,  "/a");
  BOOST_CHECK_EQUAL(update->faceId, 1);
  BOOST_CHECK_EQUAL(update->cost,   50);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);

  ++update;
  BOOST_CHECK_EQUAL(update->name,  "/a");
  BOOST_CHECK_EQUAL(update->faceId, 2);
  BOOST_CHECK_EQUAL(update->cost,   50);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);

  // Clear updates generated from previous insertions
  clearFibUpdates();

  // should generate 2 updates
  insertRoute("/a/b", 3, 0, 10, 0);

  updates = getSortedFibUpdates();
  BOOST_REQUIRE_EQUAL(updates.size(), 2);

  update = updates.begin();
  BOOST_CHECK_EQUAL(update->name,  "/a/b");
  BOOST_CHECK_EQUAL(update->faceId, 1);
  BOOST_CHECK_EQUAL(update->cost,   50);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);

  ++update;
  BOOST_CHECK_EQUAL(update->name,  "/a/b");
  BOOST_CHECK_EQUAL(update->faceId, 3);
  BOOST_CHECK_EQUAL(update->cost,   10);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);
}

BOOST_AUTO_TEST_CASE(UpdateOnLowerCostNoChildInherit)
{
  insertRoute("/", 1, 0, 50, 0);

  // Clear any updates generated from previous insertions
  clearFibUpdates();

  // Should generate 0 updates
  insertRoute("/", 1, 128, 75, 0);

  BOOST_CHECK_EQUAL(getFibUpdates().size(), 0);
}

BOOST_AUTO_TEST_CASE(UpdateOnLowerCostOnly)
{
  insertRoute("/",  1, 0, 50, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);
  insertRoute("/a", 2, 0, 10, 0);

  // Clear updates generated from previous insertions
  clearFibUpdates();

  // Should generate 2 updates: to update cost for face 1 on / and /a
  insertRoute("/", 1, 0, 25, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);

  FibUpdater::FibUpdateList updates = getSortedFibUpdates();
  BOOST_REQUIRE_EQUAL(updates.size(), 2);

  FibUpdater::FibUpdateList::const_iterator update = updates.begin();
  BOOST_CHECK_EQUAL(update->name,  "/");
  BOOST_CHECK_EQUAL(update->faceId, 1);
  BOOST_CHECK_EQUAL(update->cost,   25);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);

  ++update;
  BOOST_CHECK_EQUAL(update->name,  "/a");
  BOOST_CHECK_EQUAL(update->faceId, 1);
  BOOST_CHECK_EQUAL(update->cost,   25);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);

  // Clear updates generated from previous insertions
  clearFibUpdates();

  // Should generate 0 updates
  insertRoute("/", 1, 128, 50, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);

  BOOST_CHECK_EQUAL(getFibUpdates().size(), 0);
}

BOOST_AUTO_TEST_CASE(NoCaptureChangeWithoutChildInherit)
{
  insertRoute("/",    1, 0, 50, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);
  insertRoute("/a",   2, 0, 10, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);
  insertRoute("/a/b", 3, 0, 10, 0);
  insertRoute("/a/c", 4, 0, 10, ndn::nfd::ROUTE_FLAG_CAPTURE);

  // Clear updates generated from previous insertions
  clearFibUpdates();

  // Should generate 1 update: 1 to add face 5 to /a
  insertRoute("/a", 5, 128, 50, 0);

  const FibUpdater::FibUpdateList& updates = getFibUpdates();
  BOOST_REQUIRE_EQUAL(updates.size(), 1);

  FibUpdater::FibUpdateList::const_iterator update = updates.begin();

  BOOST_CHECK_EQUAL(update->name,  "/a");
  BOOST_CHECK_EQUAL(update->faceId, 5);
  BOOST_CHECK_EQUAL(update->cost,   50);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);
}

BOOST_AUTO_TEST_CASE(NoCaptureChangeWithChildInherit)
{
  insertRoute("/",    1, 0, 50, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);
  insertRoute("/a",   2, 0, 10, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);
  insertRoute("/a/b", 3, 0, 10, 0);
  insertRoute("/a/c", 4, 0, 10, ndn::nfd::ROUTE_FLAG_CAPTURE);

  // Clear updates generated from previous insertions
  clearFibUpdates();

  // Should generate 2 updates: one for the inserted route and
  // one to add route to /a/b
  insertRoute("/a", 4, 128, 5, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);

  FibUpdater::FibUpdateList updates = getSortedFibUpdates();
  BOOST_REQUIRE_EQUAL(updates.size(), 2);

  FibUpdater::FibUpdateList::const_iterator update = updates.begin();
  BOOST_CHECK_EQUAL(update->name,  "/a");
  BOOST_CHECK_EQUAL(update->faceId, 4);
  BOOST_CHECK_EQUAL(update->cost,   5);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);

  ++update;
  BOOST_CHECK_EQUAL(update->name,  "/a/b");
  BOOST_CHECK_EQUAL(update->faceId, 4);
  BOOST_CHECK_EQUAL(update->cost,   5);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);
}

BOOST_AUTO_TEST_CASE(CaptureTurnedOnWithoutChildInherit)
{
  insertRoute("/",    1, 0, 50, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);
  insertRoute("/a",   2, 0, 10, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);
  insertRoute("/a/b", 3, 0, 10, 0);
  insertRoute("/a/c", 4, 0, 10, 0);

  // Clear updates generated from previous insertions
  clearFibUpdates();

  // Should generate 3 updates:
  // - one for the inserted route for /a and
  // - two to remove face1 from /a/b and /a/c
  insertRoute("/a", 1, 128, 50, ndn::nfd::ROUTE_FLAG_CAPTURE);

  FibUpdater::FibUpdateList updates = getSortedFibUpdates();
  BOOST_REQUIRE_EQUAL(updates.size(), 3);

  FibUpdater::FibUpdateList::const_iterator update = updates.begin();
  BOOST_CHECK_EQUAL(update->name,  "/a");
  BOOST_CHECK_EQUAL(update->faceId, 1);
  BOOST_CHECK_EQUAL(update->cost,   50);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);

  ++update;
  BOOST_CHECK_EQUAL(update->name,  "/a/b");
  BOOST_CHECK_EQUAL(update->faceId, 1);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::REMOVE_NEXTHOP);

  ++update;
  BOOST_CHECK_EQUAL(update->name,  "/a/c");
  BOOST_CHECK_EQUAL(update->faceId, 1);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::REMOVE_NEXTHOP);
}

BOOST_AUTO_TEST_CASE(CaptureTurnedOnWithChildInherit)
{
  insertRoute("/",    1, 0, 50, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);
  insertRoute("/a",   2, 0, 10, ndn::nfd::ROUTE_FLAG_CHILD_INHERIT);
  insertRoute("/a/b", 3, 0, 10, 0);
  insertRoute("/a/c", 4, 0, 10, 0);

  // Clear updates generated from previous insertions
  clearFibUpdates();

  // Should generate 2 updates:
  // - one for the inserted route for /a and
  // - one to update /a/b with the new cost
  insertRoute("/a", 1, 128, 50, (ndn::nfd::ROUTE_FLAG_CAPTURE |
                                     ndn::nfd::ROUTE_FLAG_CHILD_INHERIT));

  FibUpdater::FibUpdateList updates = getSortedFibUpdates();
  BOOST_REQUIRE_EQUAL(updates.size(), 3);

  FibUpdater::FibUpdateList::const_iterator update = updates.begin();
  BOOST_CHECK_EQUAL(update->name,  "/a");
  BOOST_CHECK_EQUAL(update->faceId, 1);
  BOOST_CHECK_EQUAL(update->cost,   50);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);

  ++update;
  BOOST_CHECK_EQUAL(update->name,  "/a/b");
  BOOST_CHECK_EQUAL(update->faceId, 1);
  BOOST_CHECK_EQUAL(update->cost,   50);
  BOOST_CHECK_EQUAL(update->action, FibUpdate::ADD_NEXTHOP);
}

BOOST_AUTO_TEST_SUITE_END() // NewFace
BOOST_AUTO_TEST_SUITE_END() // FibUpdates
BOOST_AUTO_TEST_SUITE_END() // Rib

} // namespace nfd::tests
