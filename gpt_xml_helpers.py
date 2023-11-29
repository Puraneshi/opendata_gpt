import openai
import tiktoken

import io
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict
from pathlib import Path


"""
Adapted but mostly the same from Weixin-Liang et al.
https://github.com/Weixin-Liang/LLM-scientific-feedback/blob/main/main.py
"""
##############################################
# XML PROCESSING/PARSING
##############################################
def extract_element_text(element):
    if element.text:
        text = element.text
    else:
        text = " "
    for child in element:
        text += " " + extract_element_text(child)
        if child.tail:
            text += " " + child.tail
    return text


def get_article_title(root):
    article_title = root.find(".//article-title")
    if article_title is not None:
        title_text = article_title.text
        return title_text
    else:
        return "Artitle Title"  # not found


def get_abstract(root):
    # find the abstract element and print its text content
    abstract = root.find(".//abstract/p")
    if abstract is not None:
        return abstract.text

    abstract = root.find(".//sec[title='Abstract']")
    if abstract is not None:
        return extract_element_text(abstract)

    return "Abstract"  # not found


def get_section_text(root, section_title="Introduction"):
    """
    Warning: if introduction has subsection, it's another XML section.

    Extracts the text content of a section with the given title from the given root element.

    :param root: The root element of an XML document.
    :param section_title: The title of the section to extract. Case-insensitive.
    :return: The text content of the section as a string.
    """
    section = None
    for sec in root.findall(".//sec"):
        title_elem = sec.find("title")
        if title_elem is not None and title_elem.text.lower() == section_title.lower():
            section = sec
            break
    # If no matching section is found, return an empty string
    if section is None:
        return ""

    return extract_element_text(section)


def get_figure_and_table_captions(root):
    """
    Extracts all figure and table captions from the given root element and returns them as a concatenated string.
    """
    captions = []

    # Get Figures section
    figures = root.find('.//sec[title="Figures"]')
    if figures is not None:
        # Print Figures section content
        for child in figures:
            if child.tag == "fig":
                title = child.find("caption/title")
                caption = child.find("caption/p")
                if title is not None and title.text is not None:
                    title_text = title.text.strip()
                else:
                    title_text = ""
                if caption is not None and caption.text is not None:
                    caption_text = caption.text.strip()
                else:
                    caption_text = ""
                captions.append(f"{title_text} {caption_text}")

    # Print all table contents
    table_wraps = root.findall(".//table-wrap")
    if table_wraps is not None:
        for table_wrap in table_wraps:
            title = table_wrap.find("caption/title")
            caption = table_wrap.find("caption/p")
            if title is not None and title.text is not None:
                title_text = title.text.strip()
            else:
                title_text = ""
            if caption is not None and caption.text is not None:
                caption_text = caption.text.strip()
            else:
                caption_text = ""
            captions.append(f"{title_text} {caption_text}")

    return "\n".join(captions)


def get_main_content(root):
    """
    Get the main content of the paper, excluding the figures and tables section, usually no abstract too.

    Args:
        root: root of the xml file
    Returns:
        main_content_str: string of the main content of the paper

    """

    main_content_str = ""
    # Get all section elements
    sections = root.findall(".//sec")
    for sec in sections:  # Exclude the figures section
        # Get the section title if available
        title = sec.find("title")

        # Exclude Figures section
        if title is not None and (title.text == "Figures"):
            continue
        elif title is not None:
            main_content_str += f"\nSection Title: {title.text}\n"  # Yes, title will duplicate with extract_element_text

        main_content_str += extract_element_text(sec)
        main_content_str += "\n"

    return main_content_str


##############################################
# GPT Wrapper
# have "key.txt" with api key on project folder
##############################################
class GPTWrapper:
    def __init__(self, model_name="gpt-3.5-turbo-1106", api_key=None):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        openai.api_key = api_key or self._load_api_key

    @property
    def _load_api_key(self):
        try:
            return open("key.txt").read().strip()
        except FileNotFoundError:
            raise ValueError("API key file not found. Please provide a valid API key.")

    def make_query_args(self, user_str, n_query=1):
        system_message = {
            "role": "system",
            "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.",
        }
        user_message = {"role": "user", "content": user_str}

        query_args = {
            "model": self.model_name,
            "messages": [system_message, user_message],
            "n": n_query,
        }
        return query_args

    def compute_num_tokens(self, user_str: str) -> int:
        return len(self.tokenizer.encode(user_str))

    def send_query(self, user_str, n_query=1):
        num_tokens = self.compute_num_tokens(user_str)
        print(f"# tokens sent to GPT: {num_tokens}")

        query_args = self.make_query_args(user_str, n_query)

        try:
            completion = openai.ChatCompletion.create(**query_args)
            result = completion.choices[0]["message"]["content"]
            return result
        except openai.error.OpenAIError as e:
            print(f"Error in send_query: {e}")
            return "Error in processing the query."


# example usage
# wrapper = GPT4Wrapper(model_name="gpt-4")


def truncate(input_text: str, max_tokens: int, wrapper) -> str:
    truncated_text = wrapper.tokenizer.decode(
        wrapper.tokenizer.encode(input_text)[:max_tokens]
    )
    # Add back the closing ``` if it was truncated
    if not truncated_text.endswith("```"):
        truncated_text += "\n```"
    return truncated_text
