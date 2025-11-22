import requests
import time
import json
import statistics

def benchmark_query(url, query, query_type, iterations=3):
    print(f"\nBenchmarking {query_type} query:")
    print(f"Query: '{query}'")
    print(f"URL: {url}")
    
    latencies = []
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...", end="", flush=True)
        
        payload = {
            "query": query,
            "model": "gemini",
            "database_type": "kdb"
        }
        
        start_time = time.time()
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            print(f" Done. Latency: {latency:.4f}s")
            
            # Print complexity to verify it's working
            data = response.json()
            complexity = data.get('query_metadata', {}).get('complexity', data.get('query_complexity', 'N/A'))
            print(f"  Complexity: {complexity}")
            
        except Exception as e:
            print(f" Error: {e}")
            
    if latencies:
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        print(f"\nResults for {query_type}:")
        print(f"  Average Latency: {avg_latency:.4f}s")
        print(f"  Min Latency: {min_latency:.4f}s")
        print(f"  Max Latency: {max_latency:.4f}s")
    else:
        print("\nNo successful iterations.")

if __name__ == "__main__":
    # Wait for server to settle if needed
    print("Waiting 5 seconds for server to settle...")
    time.sleep(5)
    
    # Single-line query (baseline)
    benchmark_query(
        "http://localhost:8000/api/v1/query/v2", 
        "get me spot market data for EURUSD",
        "SINGLE_LINE",
        iterations=3
    )
    
    # Multi-line query (complex)
    benchmark_query(
        "http://localhost:8000/api/v1/query/v2",
        "calculate the 30-minute rolling correlation between EURUSD and GBPUSD for the last 7 days",
        "MULTI_LINE",
        iterations=3
    )
