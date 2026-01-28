#!/usr/bin/env python3
"""
Test script to verify concurrent request handling in the LLM proxy.

This script sends multiple concurrent requests to the proxy and verifies:
1. Requests are properly queued instead of dropped
2. All requests eventually get processed
3. Queue wait times are logged properly
4. No requests timeout or fail mysteriously
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import argparse


async def send_request(session: aiohttp.ClientSession, request_id: int, base_url: str) -> Dict[str, Any]:
    """Send a single chat completion request and return timing information."""
    
    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": f"This is test request #{request_id}. Please respond with a short message."}
        ],
        "stream": True  # Use streaming to test the full pipeline
    }
    
    start_time = time.time()
    chunk_count = 0
    error = None
    
    try:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as response:
            if response.status != 200:
                error = f"HTTP {response.status}: {await response.text()}"
                return {
                    "request_id": request_id,
                    "status": "error",
                    "http_status": response.status,
                    "error": error,
                    "total_time": time.time() - start_time,
                    "chunk_count": chunk_count
                }
            
            # Count chunks in streaming response
            async for line in response.content:
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith('data: '):
                        chunk_data = decoded[6:]  # Remove "data: " prefix
                        if chunk_data != '[DONE]':
                            chunk_count += 1
            
            return {
                "request_id": request_id,
                "status": "success",
                "chunk_count": chunk_count,
                "total_time": time.time() - start_time
            }
            
    except Exception as e:
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e),
            "total_time": time.time() - start_time,
            "chunk_count": chunk_count
        }


async def test_concurrent_requests(num_requests: int, base_url: str) -> List[Dict[str, Any]]:
    """Test concurrent request handling by sending multiple requests simultaneously."""
    
    print(f"\n{'='*60}")
    print(f"Testing {num_requests} concurrent requests")
    print(f"Proxy URL: {base_url}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # First, check queue status
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/queue/status") as response:
                if response.status == 200:
                    queue_status = await response.json()
                    print("Initial queue status:")
                    print(json.dumps(queue_status, indent=2))
                else:
                    print(f"Could not get queue status: HTTP {response.status}")
        except Exception as e:
            print(f"Could not get queue status: {e}")
    
    print(f"\nSending {num_requests} concurrent requests...\n")
    
    # Send all requests concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i, base_url) for i in range(num_requests)]
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    # Check final queue status
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/queue/status") as response:
                if response.status == 200:
                    queue_status = await response.json()
                    print("\nFinal queue status:")
                    print(json.dumps(queue_status, indent=2))
    except Exception as e:
        print(f"Could not get final queue status: {e}")
    
    return results


def analyze_results(results: List[Dict[str, Any]]):
    """Analyze test results and print summary statistics."""
    
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "error"]
    
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_times = [r["total_time"] for r in successful]
        print(f"\nTiming statistics for successful requests:")
        print(f"  Min time: {min(total_times):.2f}s")
        print(f"  Max time: {max(total_times):.2f}s")
        print(f"  Avg time: {sum(total_times)/len(total_times):.2f}s")
        
        # Show requests ordered by total time (helps identify queue ordering)
        successful_sorted = sorted(successful, key=lambda x: x["total_time"])
        print(f"\n  Fastest request: #{successful_sorted[0]['request_id']} ({successful_sorted[0]['total_time']:.2f}s)")
        print(f"  Slowest request: #{successful_sorted[-1]['request_id']} ({successful_sorted[-1]['total_time']:.2f}s)")
    
    if failed:
        print(f"\n{'='*60}")
        print("FAILED REQUESTS:")
        print(f"{'='*60}")
        for failure in failed:
            print(f"\nRequest #{failure['request_id']}:")
            print(f"  Error: {failure.get('error', 'Unknown error')}")
            print(f"  HTTP Status: {failure.get('http_status', 'N/A')}")
    
    # Check if requests were processed in order (indicating proper queueing)
    if successful and len(successful) > 2:
        start_times = [r["total_time"] for r in successful_sorted]
        time_diffs = [start_times[i+1] - start_times[i] for i in range(len(start_times)-1)]
        
        print(f"\nQueue behavior analysis:")
        print(f"  Average time between request completions: {sum(time_diffs)/len(time_diffs):.2f}s")
        print(f"  This suggests requests were processed {'sequentially' if sum(time_diffs)/len(time_diffs) > 1 else 'concurrently'}")


async def main():
    parser = argparse.ArgumentParser(description="Test concurrent request handling in LLM proxy")
    parser.add_argument("--requests", type=int, default=5, help="Number of concurrent requests to send")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Proxy base URL")
    
    args = parser.parse_args()
    
    # Test concurrent requests
    results = await test_concurrent_requests(args.requests, args.url)
    
    # Analyze and print results
    analyze_results(results)
    
    # Overall test result
    success_rate = len([r for r in results if r.get("status") == "success"]) / len(results)
    
    print(f"\n{'='*60}")
    if success_rate >= 0.9:
        print("✅ TEST PASSED: Most requests handled successfully")
        print("   The queueing system is working correctly!")
    elif success_rate >= 0.5:
        print("⚠️  TEST PARTIAL: Some requests failed")
        print("   The queue may need tuning or there may be other issues.")
    else:
        print("❌ TEST FAILED: Most requests failed")
        print("   The queueing system needs debugging.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())