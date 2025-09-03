import argparse
import sys
import csv
import requests
import json
import time
from urllib.parse import urljoin
from pathlib import Path
from bs4 import BeautifulSoup
import re


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
            urls = [line.strip().lstrip('\ufeff') for line in f if line.strip()]
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
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return None


def parse_provider_page(html_content):
    """Parse HTML content to extract structured information."""
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Extract main content sections
    sections = {}
    
    # Get the main provider information
    provider_info = {}
    
    # Try to find the main content area - the 'main' element doesn't contain provider info
    # Look for the actual content by finding where education keywords exist
    main_content = soup  # Default to full soup
    
    # Try to find a better container that actually has the provider content
    test_keywords = ["University", "Medical Degree", "Residency"]
    for container in [soup.body, soup.find('div', class_=re.compile('container', re.I)), soup]:
        if container:
            container_text = container.get_text()
            if any(keyword in container_text for keyword in test_keywords):
                main_content = container
                break
    
    # Extract name (usually in h1)
    name_elem = main_content.find('h1')
    if name_elem:
        provider_info['name'] = name_elem.get_text(strip=True)
    
    # Look for Education, Experience and Certifications section
    edu_section_content = ""
    
    # Find the education heading (can be h2, h3, h4, h5)
    edu_heading = None
    for heading in main_content.find_all(['h2', 'h3', 'h4', 'h5', 'h6']):
        heading_text = heading.get_text().lower()
        if 'education' in heading_text and ('experience' in heading_text or 'certification' in heading_text):
            edu_heading = heading
            break
        elif 'education' in heading_text:
            edu_heading = heading
            break
    
    if edu_heading:
        # Method 1: Get the parent container content (most reliable)
        parent_container = edu_heading.parent
        if parent_container:
            parent_text = parent_container.get_text(separator='\n', strip=True)
            edu_section_content = parent_text
        
        # Method 2: Also collect sibling elements with better formatting
        current = edu_heading.next_sibling
        edu_elements = []
        
        while current:
            if hasattr(current, 'name'):
                if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    # Stop when we hit another heading
                    break
                elif current.name in ['p', 'div', 'li']:
                    # Get text with proper spacing
                    text = current.get_text(separator=' ', strip=True)
                    if text and len(text) > 3:  # Skip very short text
                        edu_elements.append(text)
            elif hasattr(current, 'strip'):
                # Text node
                text = current.strip()
                if text and len(text) > 3:
                    edu_elements.append(text)
            current = current.next_sibling
        
        # Use the sibling content if it's more substantial
        sibling_content = '\n'.join(edu_elements)
        if len(sibling_content) > len(edu_section_content):
            edu_section_content = sibling_content
        
        sections['education_section'] = edu_section_content
    
    # Also look for provider details sections that might contain education info
    provider_details = main_content.find('div', class_=re.compile('provider.*detail', re.I))
    if provider_details:
        sections['provider_details'] = provider_details.get_text(separator='\n', strip=True)
    
    # Get all text content for fallback
    sections['full_text'] = main_content.get_text(separator='\n', strip=True)
    
    # Look for last modified date (try multiple approaches)
    last_modified = ""
    
    # Method 1: Look in footer
    footer = soup.find('footer') or soup.find('div', class_=re.compile('footer', re.I))
    if footer:
        footer_text = footer.get_text()
        date_match = re.search(r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\w+ \d{1,2}, \d{4})', footer_text)
        if date_match:
            last_modified = date_match.group(1)
    
    # Method 2: Look for "Last updated" or "Modified" text anywhere in full page
    if not last_modified:
        full_text = soup.get_text()  # Use full page text, not just main_content
        patterns = [
            r'last modified[,:\s]+([A-Za-z]+ \d{1,2}, \d{4})',  # "Last Modified, July 25, 2024"
            r'last updated[,:\s]+([A-Za-z]+ \d{1,2}, \d{4})',   # "Last Updated, July 25, 2024"
            r'modified[,:\s]+([A-Za-z]+ \d{1,2}, \d{4})',      # "Modified, July 25, 2024"
            r'updated[,:\s]+([A-Za-z]+ \d{1,2}, \d{4})',       # "Updated, July 25, 2024"
            r'last modified[,:\s]+(\d{1,2}/\d{1,2}/\d{4})',    # MM/DD/YYYY format
            r'last modified[,:\s]+(\d{4}-\d{2}-\d{2})'         # YYYY-MM-DD format
        ]
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                last_modified = match.group(1)
                break
    
    return sections, provider_info, last_modified


def extract_provider_data(content, url, api_endpoint, model, api_key):
    """Extract structured data from provider profile using LLM API."""
    
    # Parse the HTML content first to get structured sections
    sections, provider_info, last_modified = parse_provider_page(content)
    
    # Build focused content for the LLM - use multiple approaches
    focused_content = ""
    
    # If we found a specific education section, prioritize it
    if sections.get('education_section') and len(sections['education_section']) > 50:
        focused_content += "=== EDUCATION, EXPERIENCE AND CERTIFICATIONS SECTION ===\n"
        focused_content += sections['education_section'] + "\n\n"
    
    # If we found provider details, include them
    if sections.get('provider_details') and len(sections['provider_details']) > 50:
        focused_content += "=== PROVIDER DETAILS SECTION ===\n"
        focused_content += sections['provider_details'] + "\n\n"
    
    # Always include full content as fallback, but with generous limits to capture all sections
    focused_content += "=== FULL PAGE CONTENT ===\n"
    # Use larger content limit to ensure we capture Provider Background, Diseases Treated, etc.
    content_limit = 12000  # Increased limit to capture all sections
    focused_content += sections['full_text'][:content_limit]
    
    prompt = f"""
You are extracting information from a Fred Hutchinson Cancer Center provider profile page. Please extract ALL the following information and return it as valid JSON:

{{
  "Name": "Full name of the provider",
  "Credentials": "Professional credentials after name (MD, PhD, MPH, etc.)",
  "Titles": "Professional titles and positions",
  "Specialty": "Medical specialty/specialties",
  "Locations": "Practice locations and clinics", 
  "Areas of Clinical Practice": "Clinical practice areas and focus - look under 'Provider Background' section",
  "Diseases Treated": "All diseases and conditions treated - look for 'Diseases Treated' section with specific disease names",
  "Languages": "Languages spoken",
  "Undergraduate Degree": "Undergraduate education and institution",
  "Medical Degree": "Medical school and degree",
  "Residency": "Residency training program and institution",
  "Fellowship": "Fellowship training and specialization",
  "Board Certifications": "Medical board certifications with dates if available",
  "Awards": "Awards, honors, recognition, and achievements",
  "Other": "Other relevant information like MPH, internships, additional training, research experience"
}}

CRITICAL INSTRUCTIONS:
1. EXTRACT ALL FIELDS - Search the ENTIRE content thoroughly
2. For "Areas of Clinical Practice" - Look specifically for the "Provider Background" section, then find "Area of Clinical Practice" subsection
3. For "Diseases Treated" - Look for the "Diseases Treated" section which lists specific diseases/conditions in a structured format
4. For "Credentials" - Extract degree abbreviations that appear after the provider name (MD, MPH, PhD, etc.)
5. For "Awards" - look for any recognition, top doctor awards, honors mentioned
6. For "Other" - include MPH details, post-doctoral fellowships, research experience, internships
7. Look in "Education, Experience and Certifications" section for degrees/training
8. Pay special attention to section headings like "Provider Background", "Diseases Treated", "Area of Clinical Practice"
9. Return ONLY valid JSON - no extra text or explanation
10. Use empty string "" ONLY if information truly cannot be found after thorough search

Provider profile content:
{focused_content}
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
        llm_content = result["choices"][0]["message"]["content"]
        
        # Try to parse JSON from the response
        try:
            # Look for JSON in the response
            start_idx = llm_content.find('{')
            end_idx = llm_content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = llm_content[start_idx:end_idx]
                data = json.loads(json_str)
                
                # Add the profile URL and last modified date
                data["Profile URL"] = url
                if last_modified:
                    data["Last Modified"] = last_modified
                elif "Last Modified" not in data:
                    data["Last Modified"] = ""
                
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
