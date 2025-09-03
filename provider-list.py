import argparse
import sys
import csv
import requests
import json
import time
from urllib.parse import urljoin
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create CSV inventory of Fred Hutchinson & UW Medicine provider profiles"
    )
    
    # Required arguments
    parser.add_argument(
        "url_file",
        help="Text file containing provider URLs (one per line)"
    )
    parser.add_argument(
        "output_csv",
        help="Output CSV file name"
    )
    
    # Optional arguments
    parser.add_argument(
        "--endpoint",
        default="http://localhost:11434/v1/chat/completions",
        help="OpenAI-compatible API endpoint URL (default: %(default)s)"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:3b",
        help="LLM model name (default: %(default)s)"
    )
    parser.add_argument(
        "--api-key",
        default="sk-1234",
        help="API key for the LLM endpoint (default: %(default)s)"
    )
    
    return parser.parse_args()


def load_urls(url_file):
    """Load URLs from text file, one per line."""
    try:
        with open(url_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls
    except FileNotFoundError:
        print(f"Error: URL file '{url_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading URL file '{url_file}': {e}")
        sys.exit(1)


def fetch_page_content(url):
    """Fetch content from a provider profile URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return None


def extract_provider_data(content, url, api_endpoint, model, api_key):
    """Extract structured data from provider profile using LLM API."""
    prompt = f"""
Please extract the following information from this Fred Hutchinson Cancer Center provider profile page and return it as JSON:

{{
  "Name": "Full name",
  "Credentials": "Professional credentials (MD, PhD, etc.)",
  "Titles": "Professional titles and positions",
  "Specialty": "Medical specialty/specialties",
  "Locations": "Practice locations",
  "Areas of Clinical Practice": "Clinical practice areas",
  "Diseases Treated": "Diseases and conditions treated",
  "Languages": "Languages spoken",
  "Undergraduate Degree": "Undergraduate education",
  "Medical Degree": "Medical school and degree",
  "Residency": "Residency training",
  "Fellowship": "Fellowship training",
  "Board Certifications": "Board certifications",
  "Awards": "Awards and recognition",
  "Other": "Other relevant information",
  "Last Modified": "Last modified date from page footer, store in YYYY-MM-DD format"
}}

Extract only the information that is present. Use empty string if information is not available.

Provider profile content:
{content[:10000]}  # Limit content to avoid token limits
"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Try to parse JSON from the response
        try:
            # Look for JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                data = json.loads(json_str)
                data["Profile URL"] = url
                return data
            else:
                return None
        except json.JSONDecodeError:
            return None
            
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None


def write_csv_header(output_file):
    """Write CSV header row."""
    fieldnames = [
        "Name", "Credentials", "Titles", "Specialty", "Locations",
        "Areas of Clinical Practice", "Diseases Treated", "Languages",
        "Undergraduate Degree", "Medical Degree", "Residency", "Fellowship",
        "Board Certifications", "Awards", "Other", "Profile URL", "Last Modified"
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return fieldnames


def append_to_csv(output_file, data, fieldnames):
    """Append a row to the CSV file."""
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Ensure all fields are present
        row = {field: data.get(field, "") for field in fieldnames}
        writer.writerow(row)


def print_progress(current, total, url, success):
    """Print progress information."""
    percentage = (current / total) * 100
    status = "✓" if success else "✗"
    print(f"[{current:3d}/{total:3d}] ({percentage:5.1f}%) {status} {url}")


def main():
    args = parse_arguments()
    
    # Load URLs
    print(f"Loading URLs from {args.url_file}...")
    urls = load_urls(args.url_file)
    print(f"Found {len(urls)} URLs to process")
    
    # Initialize CSV file
    print(f"Initializing output CSV: {args.output_csv}")
    fieldnames = write_csv_header(args.output_csv)
    
    # Process URLs
    successful = 0
    failed_urls = []
    
    print("\nProcessing provider profiles...")
    print("=" * 70)
    
    for i, url in enumerate(urls, 1):
        # Fetch page content
        content = fetch_page_content(url)
        
        if content is None:
            failed_urls.append(url)
            print_progress(i, len(urls), url, False)
            continue
        
        # Extract data using LLM
        provider_data = extract_provider_data(content, url, args.endpoint, args.model, args.api_key)
        
        if provider_data is None:
            failed_urls.append(url)
            print_progress(i, len(urls), url, False)
            continue
        
        # Write to CSV
        append_to_csv(args.output_csv, provider_data, fieldnames)
        successful += 1
        print_progress(i, len(urls), url, True)
        
        # Small delay
        time.sleep(0.5)
    
    # Print final statistics
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total URLs processed: {len(urls)}")
    print(f"Successful extractions: {successful}")
    print(f"Failed extractions: {len(failed_urls)}")
    print(f"Success rate: {(successful/len(urls)*100):.1f}%")
    print(f"Output written to: {args.output_csv}")
    
    if failed_urls:
        print("\nFailed URLs:")
        for url in failed_urls:
            print(f"  - {url}")


if __name__ == "__main__":
    main()
