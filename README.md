# Open Data with GPT

## Overview

This project provides a set of Python scripts for analyzing scientific papers using XML data extraction and generating language model reviews. It includes functions for parsing XML documents related to scientific papers and a GPT (Generative Pre-trained Transformer) wrapper for querying OpenAI's GPT API.

**Adapted from the work of [Weixin-Liang et al.](https://github.com/Weixin-Liang/LLM-scientific-feedback/)**

## Features

- **XML Processing:**
  - Functions for extracting information from scientific papers in XML format, including title, abstract, introduction, figure and table captions, section titles, and main content.

- **GPT Wrapper:**
  - A wrapper class for interacting with GPT API.
  - Tokenization and API calls for generating language model reviews based on parsed paper information.

- **Workflow Steps:**
  - Functions for creating a truncated prompt for the GPT model based on parsed XML data.
  - Steps for evaluating whether a scientific paper includes information on obtaining reported data.
  - Task-specific questions and CSV formatting instructions for user interaction.

## Getting Started

1. **Dependencies:**
   - Install required Python libraries (`openai`, `tiktoken`, etc.). Ensure the API key is available in a file named "key.txt" in the project folder.

2. **Example Usage:**
   - Use provided functions like `step2_parse_xml` to extract information from XML.
   - Generate language model reviews with `step3_get_lm_review`.

3. **Customization:**
   - Adapt the number of tokens and modify functions as needed for specific use cases.

## Usage Example

```python
# Example usage of XML parsing and language model review
parsed_xml = step2_parse_xml(xml_data)
review_result = step3_get_lm_review(gpt_wrapper, parsed_xml)
print(review_result)
```

## License

This project is licensed under the [MIT License](LICENSE).

---
