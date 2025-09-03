# Provider List Creator

A utility that creates a CSV inventory of Fred Hutchinson & UW Medicine medical providers by scraping their profile webpages and extracting structured data using a Large Language Model (LLM).

## Overview

This tool automates the process of collecting detailed information about medical providers from the Fred Hutchinson Cancer Center website. It processes a list of provider profile URLs, scrapes the content from each page, and uses an OpenAI-compatible LLM API to extract structured data into a standardized CSV format.

## How It Works

1. **URL Processing**: Reads a text file containing provider profile URLs (one per line)
2. **Web Scraping**: Fetches HTML content from each provider profile page
3. **Data Extraction**: Sends the HTML content to an LLM API to extract structured information
4. **CSV Generation**: Writes extracted data to a CSV file with standardized columns
5. **Progress Tracking**: Displays real-time progress and statistics during processing
6. **Error Handling**: Tracks failed URLs and provides comprehensive reporting

## Installation

This project uses `uv` for Python package management. To set up the project:

```bash
# Clone or download the repository
cd provider-list

# Install dependencies
uv sync
```

## Usage

### Basic Usage

```bash
uv run provider-list.py <url_file> <output_csv>
```

### Example

```bash
# Process all provider URLs and create CSV
uv run provider-list.py provider-urls.txt fredhutch_providers.csv

# Test with a smaller subset
head -10 provider-urls.txt > test-urls.txt
uv run provider-list.py test-urls.txt test-output.csv
```

### Advanced Usage

The utility accepts several optional parameters for customizing the LLM API connection:

Example using with LiteLLM + AWS Bedrock + Claude Sonnet:

```bash
uv run provider-list.py   provider-test-urls.txt fh-providers-2025-09-03-test7.csv \
  --endpoint "http://localhost:4000/v1/chat/completions" \
  --model claude-4-sonnet \
  --api-key sk-1234
```

## Command Line Options

- `url_file` (required): Text file containing provider URLs, one per line
- `output_csv` (required): Name of the output CSV file to create
- `--endpoint`: OpenAI-compatible API endpoint URL (default: `http://localhost:11434/v1/chat/completions`)
- `--model`: LLM model name (default: `qwen2.5:3b`)
- `--api-key`: API key/bearer token for the LLM endpoint (default: `sk-1234`)

## Output Format

The utility generates a CSV file with the following columns:

- Name
- Credentials 
- Titles
- Specialty
- Locations
- Areas of Clinical Practice
- Diseases Treated
- Languages
- Undergraduate Degree
- Medical Degree
- Residency
- Fellowship
- Board Certifications
- Awards
- Other
- Profile URL
- Last Modified

## Progress Tracking

During execution, the utility displays:
- Current progress (X/Y URLs processed)
- Percentage complete
- Success/failure status for each URL
- Final statistics including success rate and failed URLs

## Error Handling

The utility handles various error conditions:
- Network timeouts and connection errors
- Invalid or unreachable URLs
- LLM API failures
- JSON parsing errors
- File I/O errors

Failed URLs are tracked and reported at the end of processing.

## Requirements

- Python 3.12+
- Internet connection for web scraping and LLM API calls
- Access to an OpenAI-compatible LLM API endpoint
- Input file with valid Fred Hutchinson provider profile URLs

## LLM Setup

This utility is designed to work with local LLM servers like Ollama. To set up Ollama:

1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull qwen2.5:3b`
3. The API will be available at `http://localhost:11434/v1/chat/completions`

You can also use other OpenAI-compatible APIs by adjusting the endpoint, model, and API key parameters.