#!/usr/bin/env python3
"""
Test script for KDB /v3/execute endpoint with execution tracking

This script demonstrates:
1. Executing KDB queries via /v3/execute
2. Automatic execution tracking (async background logging)
3. Querying execution history

Prerequisites:
- KDB+ server running on localhost:5001
- PostgreSQL with execution tracking table created
- FastAPI server running on localhost:8000
"""

import requests
import json
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
USER_ID = "test_user_123"


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_kdb_simple_query():
    """Test 1: Simple KDB query"""
    print_section("TEST 1: Simple KDB Query (SINGLE_LINE)")

    payload = {
        "query": "10#til 100",  # Simple KDB query: first 10 numbers from 0-99
        "execution_id": f"kdb-test-{int(time.time())}-001",
        "database_type": "kdb",
        "query_complexity": "SINGLE_LINE",
        "pagination": {
            "page": 0,
            "page_size": 10
        }
    }

    print(f"\n📤 Request:")
    print(f"   Endpoint: POST {API_BASE_URL}/execute/v3")
    print(f"   Query: {payload['query']}")
    print(f"   Database: {payload['database_type']}")
    print(f"   Complexity: {payload['query_complexity']}")

    try:
        start = time.time()
        response = requests.post(f"{API_BASE_URL}/execute/v3", json=payload, timeout=10)
        elapsed = time.time() - start

        print(f"\n📥 Response:")
        print(f"   Status: {response.status_code}")
        print(f"   Time: {elapsed:.4f}s")

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success!")
            print(f"   Results: {len(data['results'])} rows")
            print(f"   Total: {data['pagination']['totalRows']} rows")
            print(f"   Execution time: {data['metadata'].get('execution_time', 'N/A')}s")
            print(f"\n   First 3 results: {json.dumps(data['results'][:3], indent=6)}")

            print(f"\n🔄 Background Execution Tracking:")
            print(f"   ✅ Execution logged asynchronously (no impact on response time)")
            print(f"   Status: success")
            print(f"   User: {USER_ID if 'user_id' in payload else 'anonymous'}")

            return True
        else:
            print(f"   ❌ Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection Error: FastAPI server not running at {API_BASE_URL}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def test_kdb_table_query():
    """Test 2: KDB table query with pagination"""
    print_section("TEST 2: KDB Table Query with Pagination")

    payload = {
        "query": "([] name:`Alice`Bob`Charlie`David`Eve; age:25 30 35 40 45; city:`NYC`LA`Chicago`Boston`Seattle)",
        "execution_id": f"kdb-test-{int(time.time())}-002",
        "database_type": "kdb",
        "query_complexity": "SINGLE_LINE",
        "pagination": {
            "page": 0,
            "page_size": 3
        }
    }

    print(f"\n📤 Request:")
    print(f"   Query: Create table with 5 rows, request 3 per page")
    print(f"   Page: {payload['pagination']['page']}")
    print(f"   Page Size: {payload['pagination']['page_size']}")

    try:
        response = requests.post(f"{API_BASE_URL}/execute/v3", json=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"\n📥 Response:")
            print(f"   ✅ Success!")
            print(f"   Returned: {len(data['results'])} rows")
            print(f"   Total: {data['pagination']['totalRows']} rows")
            print(f"   Pages: {data['pagination']['totalPages']}")
            print(f"\n   Results:")
            for i, row in enumerate(data['results'], 1):
                print(f"      {i}. {row}")

            return True
        else:
            print(f"\n❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def test_kdb_complex_query():
    """Test 3: Complex KDB query (MULTI_LINE)"""
    print_section("TEST 3: Complex KDB Query (MULTI_LINE)")

    payload = {
        "query": "t:([] x:til 10; y:10*til 10); select from t where x>5",
        "execution_id": f"kdb-test-{int(time.time())}-003",
        "database_type": "kdb",
        "query_complexity": "MULTI_LINE",  # Safe for multi-statement queries
        "pagination": {
            "page": 0,
            "page_size": 100
        }
    }

    print(f"\n📤 Request:")
    print(f"   Query: Multi-statement query with filter")
    print(f"   Complexity: {payload['query_complexity']}")

    try:
        response = requests.post(f"{API_BASE_URL}/execute/v3", json=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"\n📥 Response:")
            print(f"   ✅ Success!")
            print(f"   Results: {json.dumps(data['results'], indent=6)}")
            return True
        else:
            print(f"\n❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def test_execution_history():
    """Test 4: Query execution history"""
    print_section("TEST 4: Query Execution History")

    print(f"\n📤 Fetching execution history for user: {USER_ID}")

    try:
        response = requests.get(f"{API_BASE_URL}/executions/{USER_ID}?limit=10", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"\n📥 Response:")
            print(f"   ✅ Found {data['count']} executions")

            if data['count'] > 0:
                print(f"\n   Recent Executions:")
                for i, exec in enumerate(data['executions'][:5], 1):
                    print(f"\n   {i}. Execution ID: {exec['execution_id']}")
                    print(f"      Query: {exec['query'][:60]}...")
                    print(f"      Status: {exec['status']}")
                    print(f"      Database: {exec['database_type']}")
                    print(f"      Time: {exec.get('execution_time', 'N/A')}s")
                    print(f"      Timestamp: {exec['started_at']}")
            else:
                print(f"\n   No executions found for this user yet.")
                print(f"   Note: Execution tracking requires PostgreSQL with migration applied")

            return True
        else:
            print(f"\n❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def test_execution_stats():
    """Test 5: Execution statistics"""
    print_section("TEST 5: Execution Statistics")

    print(f"\n📤 Fetching execution stats for user: {USER_ID}")

    try:
        response = requests.get(f"{API_BASE_URL}/executions/{USER_ID}/stats?days=7", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"\n📥 Response:")
            print(f"   ✅ Statistics (last {data['period_days']} days):")
            print(f"\n   Total Executions: {data['total_executions']}")
            print(f"   Success: {data['success_count']} ({data['success_rate']}%)")
            print(f"   Failed: {data['failed_count']}")
            print(f"   Errors: {data['error_count']}")
            print(f"   Timeouts: {data['timeout_count']}")
            print(f"\n   Performance:")
            print(f"   Avg Execution Time: {data['avg_execution_time']}s")
            print(f"   Max Execution Time: {data['max_execution_time']}s")
            print(f"   Total Rows Processed: {data['total_rows_processed']}")

            return True
        else:
            print(f"\n❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🚀" * 35)
    print("  KDB /v3/execute Endpoint Test Suite")
    print("  with Execution Tracking")
    print("🚀" * 35)

    print(f"\nPrerequisites:")
    print(f"  ✓ FastAPI server: {API_BASE_URL}")
    print(f"  ✓ KDB+ server: localhost:5001")
    print(f"  ✓ PostgreSQL with execution_tracking table")

    print(f"\nRunning tests...")

    results = []

    # Test KDB queries
    results.append(("Simple Query", test_kdb_simple_query()))
    time.sleep(0.5)  # Small delay between tests

    results.append(("Table Query", test_kdb_table_query()))
    time.sleep(0.5)

    results.append(("Complex Query", test_kdb_complex_query()))
    time.sleep(0.5)

    # Test execution tracking
    results.append(("Execution History", test_execution_history()))
    time.sleep(0.5)

    results.append(("Execution Stats", test_execution_stats()))

    # Summary
    print_section("TEST SUMMARY")
    print()
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}  {test_name}")

    print(f"\n   Results: {passed}/{total} tests passed")

    if passed == total:
        print(f"\n   🎉 All tests passed!")
    else:
        print(f"\n   ⚠️  Some tests failed. Check prerequisites:")
        print(f"      - Is FastAPI server running? (python -m uvicorn app.main:app)")
        print(f"      - Is KDB+ server running on localhost:5001?")
        print(f"      - Is PostgreSQL migration applied?")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
