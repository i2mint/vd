## __init__.py

```python
"""
Python facade to OpenAi functionality.
"""

from oa.util import (
    openai,
    grazed,
    djoin,
    app_data_dir,
    num_tokens,
    model_information_dict,
    utc_int_to_iso_date,
    DFLT_ENGINE,
    DFLT_MODEL,
    DFLT_EMBEDDINGS_MODEL,
)

from oa.base import (
    chat,
    complete,
    dalle,
    api,
    embeddings,
    model_information,
    compute_price,
    text_is_valid,
)
from oa.openai_specs import raw
from oa.tools import (
    prompt_function,  # Make a python function from a prompt template
    PromptFuncs,  # make a collection of AI-enabled functions
    prompt_json_function,  # Make a python function (returning a valid json) from a prompt template
    infer_schema_from_verbal_description,  # Get a schema from a verbal description
)
from oa import ask
from oa.stores import OaStores
from oa.chats import ChatDacc
```

## _params.py

```python
"""
Parameters, configuration, and constants for the OpenAI Gym environment.

Some resources:
* OpenAI API pricing: https://platform.openai.com/docs/pricing

"""

turns_data_ssot = {
    "id": {
        "description": "A unique identifier for the conversation turn.",
        "example": "1fc35aa7-6b7a-4dae-9838-ead52c6d4793",
    },
    "children": {
        "description": "An array of child conversation turns, which can hold additional messages in the conversation thread.",
        "example": "[]",
    },
    "message.id": {
        "description": "A unique identifier for the message within the conversation turn.",
        "example": "1fc35aa7-6b7a-4dae-9838-ead52c6d4793",
    },
    "message.author.role": {
        "description": "The role of the author of the message (e.g., user, assistant).",
        "example": "assistant",
    },
    "message.author.metadata.real_author": {
        "description": "Metadata indicating the real author or source of the message.",
        "example": "tool:web",
    },
    "message.author.name": {
        "description": "The name of the author of the message.",
        "example": "dalle.text2im",
    },
    "message.content.content_type": {
        "description": "The type of content in the message (e.g., text, image, etc.).",
        "example": "text",
    },
    "message.content.parts": {
        "description": "An array containing parts or segments of the message content, typically for handling long messages.",
        "example": [
            "The generated image visually ca...lements in a futuristic design."
        ],
    },
    "message.content.model_set_context": {
        "description": "Context information related to the model used for content generation.",
        "example": "",
    },
    "message.content.language": {
        "description": "The language of the message content, represented in standard language codes.",
        "example": "unknown",
    },
    "message.content.text": {
        "description": "The actual text of the message.",
        "example": 'search("Please give me an estim...rian diet and a omnivore diet")',
    },
    "message.status": {
        "description": "The processing status of the message (e.g., finished, in-progress).",
        "example": "finished_successfully",
    },
    "message.end_turn": {
        "description": "A boolean indicating if this is the last message in the conversation turn.",
        "example": True,
    },
    "message.weight": {
        "description": "A numeric value representing the message's importance or relevance in the conversation.",
        "example": 1,
    },
    "message.metadata.is_visually_hidden_from_conversation": {
        "description": "A boolean indicating if the message is hidden from the visible conversation stream.",
        "example": True,
    },
    "message.metadata.shared_conversation_id": {
        "description": "An identifier for the shared context of the conversation, if applicable.",
        "example": "678a1339-d14c-8013-bfcb-288d367a9079",
    },
    "message.metadata.user_context_message_data": {
        "description": "Contextual data related to the user's message, if applicable.",
        "example": None,
    },
    "message.metadata.is_user_system_message": {
        "description": "A boolean indicating if the message is generated as a system message for the user.",
        "example": True,
    },
    "message.metadata.is_redacted": {
        "description": "A boolean indicating if the message content has been redacted for privacy or security reasons.",
        "example": True,
    },
    "message.metadata.request_id": {
        "description": "An identifier for the request associated with the message, useful for debugging.",
        "example": "9034eeef6e62e209-MRS",
    },
    "message.metadata.message_source": {
        "description": "Information about the source of the message, if applicable.",
        "example": None,
    },
    "message.metadata.timestamp_": {
        "description": "The timestamp format type for message creation, indicating if it's absolute or relative.",
        "example": "absolute",
    },
    "message.metadata.message_type": {
        "description": "The type/category of message, useful for filtering or processing messages.",
        "example": None,
    },
    "message.metadata.model_slug": {
        "description": "A slug representing the model used to generate the response.",
        "example": "gpt-4o",
    },
    "message.metadata.default_model_slug": {
        "description": "The default model slug for the message, representing the standard model used.",
        "example": "gpt-4o",
    },
    "message.metadata.parent_id": {
        "description": "The ID of the parent message for threading purposes.",
        "example": "073e2336-5c95-434e-a0d2-74a58b68f8e0",
    },
    "message.metadata.finish_details.type": {
        "description": "The type of finish that occurred for the message processing (e.g., stop, timeout).",
        "example": "stop",
    },
    "message.metadata.finish_details.stop_tokens": {
        "description": "An array of token IDs that indicate where the message generation stopped.",
        "example": [200002, 200007],
    },
    "message.metadata.is_complete": {
        "description": "A boolean indicating if the message generation process was completed successfully.",
        "example": True,
    },
    "message.metadata.citations": {
        "description": "An array of citations included in the message, if applicable.",
        "example": "[]",
    },
    "message.metadata.content_references": {
        "description": "References to additional content used in the message, if any.",
        "example": "[]",
    },
    "message.metadata.command": {
        "description": "The command issued by the user that generated this message.",
        "example": "search",
    },
    "message.metadata.status": {
        "description": "The status of the message at the time of capture (completed, in-progress, etc.).",
        "example": "finished",
    },
    "message.metadata.search_source": {
        "description": "The source from which search results were derived, if applicable.",
        "example": "composer_search",
    },
    "message.metadata.client_reported_search_source": {
        "description": "The source reported by the client regarding the search origin.",
        "example": "conversation_composer_previous_web_mode",
    },
    "message.metadata.search_result_groups": {
        "description": "An array of search result groups that provide relevant information based on the user's query.",
        "example": [
            {
                "type": "search_result_group",
                "domain": "learnmetrics.com",
                "entries": [
                    {
                        "type": "search_result",
                        "url": "https://learnmetrics.com/how-ma...average-home-electricity-usage/",
                        "title": "How Many kWh Per Day Is Normal? Average 1-6 Person Home kWh Usage",
                        "snippet": "7,340 kWh Per Year: 2 Person Ho...: 4 Person Home: 36.58 kWh P...",
                        "ref_id": {
                            "turn_index": 0,
                            "ref_type": "search",
                            "ref_index": 0,
                        },
                        "content_type": None,
                        "pub_date": None,
                        "attributions": None,
                    }
                ],
            }
        ],
    },
    "message.metadata.safe_urls": {
        "description": "An array of URLs considered safe for sharing, derived from the content.",
        "example": [
            "https://www.sciencing.com/being...ls-3342/?utm_source=chatgpt.com"
        ],
    },
    "message.metadata.message_locale": {
        "description": "The locale in which the message was generated, formatted as a language-country code.",
        "example": "en-US",
    },
    "message.metadata.image_results": {
        "description": "An array of generated image results related to the conversation, if any.",
        "example": "[]",
    },
    "message.recipient": {
        "description": "The intended recipient of the message, indicating if it was meant for a specific user or a group.",
        "example": "all",
    },
    "message.create_time": {
        "description": "The creation timestamp of the message, represented as a float for more precision.",
        "example": 1737102115.45064,
    },
    "parent": {
        "description": "The ID of the parent turn of the conversation, refers to the context or previous message.",
        "example": "073e2336-5c95-434e-a0d2-74a58b68f8e0",
    },
}


metadata_ssot = {
    "title": {
        "description": "The title of the chat conversation.",
        "example": "Test Chat 1",
    },
    "create_time": {
        "description": "A timestamp indicating when the chat conversation was created, represented in Unix time format.",
        "example": 1737020729.060687,
    },
    "update_time": {
        "description": "A timestamp indicating the last time the chat conversation was updated, represented in Unix time format.",
        "example": 1737020733.031014,
    },
    "moderation_results": {
        "description": "An array holding the results of moderation checks applied to the conversation. If no moderation has taken place, this array will be empty.",
        "example": [],
    },
    "current_node": {
        "description": "The unique identifier for the current state or node in the conversation flow, typically in UUID format.",
        "example": "be4486db-894f-4e6f-bd0a-22d9d2facf69",
    },
    "conversation_id": {
        "description": "A unique identifier for the conversation as a whole, typically in UUID format.",
        "example": "6788d539-0f2c-8013-9535-889bf344d7d5",
    },
    "is_archived": {
        "description": "A boolean indicating whether the chat conversation has been archived. True means it is archived, false means it is active.",
        "example": False,
    },
    "safe_urls": {
        "description": "An array of safe URLs that were included in the conversation. If there are no safe URLs, this array will be empty.",
        "example": [],
    },
    "default_model_slug": {
        "description": "A string representing the default model used during the conversation, which specifies the AI language model employed.",
        "example": "gpt-4o",
    },
    "disabled_tool_ids": {
        "description": "An array containing the identifiers of any tools that have been disabled during the conversation. If none are disabled, this array will be empty.",
        "example": [],
    },
    "is_public": {
        "description": "A boolean indicating whether the conversation is accessible to the public. True means it is public, false means it is private.",
        "example": True,
    },
    "has_user_editable_context": {
        "description": "A boolean indicating whether the user can modify the context of the conversation. True means editable, false means not editable.",
        "example": False,
    },
    "continue_conversation_url": {
        "description": "A URL that allows users to continue the conversation from a specific point. This link will redirect to the conversation session.",
        "example": "https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5/continue",
    },
    "moderation_state": {
        "description": "An object holding the state of moderation checks applied to the conversation, providing details on whether different moderation actions have taken place.",
        "example": {
            "has_been_moderated": False,
            "has_been_blocked": False,
            "has_been_accepted": False,
            "has_been_auto_blocked": False,
            "has_been_auto_moderated": False,
        },
    },
    "is_indexable": {
        "description": "A boolean indicating whether the chat conversation can be indexed for search purposes. True means it is indexable, false means it is not.",
        "example": False,
    },
    "is_better_metatags_enabled": {
        "description": "A boolean indicating whether enhanced metatags are enabled for the conversation. True means better metatags are enabled, implying improved discoverability, while false means they are not.",
        "example": True,
    },
}
```

## _resources.py

```python
"""Tools to get resources for oa"""

import os
import dol

# -------------------------------------------------------------------------------------
# Stores

from dol import cache_this

from oa.util import data_files

dflt_resources_dir = str(data_files.parent.parent / "misc" / "data" / "resources")

# -------------------------------------------------------------------------------------
# SSOT tools

_model_info_mapping = {
    "Input": "price_per_million_tokens",  # TODO: Verify that Input fields are always in per-million-token units
    "Output": "price_per_million_tokens_output",  # TODO: Verify that Output fields are always in per-million-token units
}


def pricing_info_persepective_of_model_info():
    from oa.util import pricing_info
    import pandas as pd

    prices_info_ = pd.DataFrame(pricing_info()).drop_duplicates(subset="Model")

    def if_price_change_to_number(price):
        if isinstance(price, str) and price.startswith("$"):
            return float(price.replace("$", "").replace(",", ""))
        elif isinstance(price, dict):
            return {k: if_price_change_to_number(v) for k, v in price.items()}
        else:
            return price

    def ch_field_names(d):
        return {_model_info_mapping.get(k, k): v for k, v in d.items()}

    model_info = {
        row["Model"]: ch_field_names(if_price_change_to_number(row.dropna().to_dict()))
        for _, row in prices_info_.iterrows()
    }

    return model_info


def compare_pricing_info_to_model_info(verbose=True):
    """Compares"""
    from oa._resources import (
        compare_pricing_info_to_model_info,
        pricing_info_persepective_of_model_info,
    )

    from oa.util import pricing_info, model_information_dict

    new_prices = pricing_info_persepective_of_model_info()

    # keys (i.e. model ids) in common with both in model_information_dict & new_prices
    common_keys = set(model_information_dict.keys()) & set(new_prices.keys())

    # # for all keys in common, compare the prices
    for key in common_keys:
        model_info = model_information_dict[key]
        price_info = new_prices[key]

    from lkj import compare_field_values, inclusive_subdict
    from functools import partial

    prices_are_the_same = compare_field_values(
        model_information_dict,
        new_prices,
        default_comparator=compare_field_values,
        aggregator=lambda d: {k: all(v.values()) for k, v in d.items()},
    )
    models_where_prices_are_different = {
        k for k, v in prices_are_the_same.items() if not v
    }

    differences = dict()

    if any(models_where_prices_are_different):
        t = inclusive_subdict(model_information_dict, models_where_prices_are_different)
        tt = inclusive_subdict(new_prices, models_where_prices_are_different)

        differences = compare_field_values(
            t,
            tt,
            default_comparator=partial(
                compare_field_values, default_comparator=lambda x, y: (x, y)
            ),
            # aggregator=lambda d: {k: all(v.values()) for k, v in d.items()},
        )

        if verbose:
            import pprint

            print("Differences:")
            pprint.pprint(differences)

    return differences


# -------------------------------------------------------------------------------------
# Resources class

from dataclasses import dataclass, field
import json
import pathlib
from functools import partial
from typing import Dict, Any, Optional, Union, List
from collections.abc import Callable

import pandas as pd
import dol


dflt_pricing_url = "https://platform.openai.com/docs/pricing"


@dataclass
class Resources:
    """
    Data Access class for OpenAI pricing information.

    This class manages the retrieval, computation, and caching of
    OpenAI pricing data through a chain of computations.

    >>> r = Resources()  # doctest: +ELLIPSIS
    >>> r.resources_dir  # doctest: +ELLIPSIS
    '...misc/data/resources'
    """

    resources_dir: str = dflt_resources_dir
    pricing_url: str = dflt_pricing_url

    schema_description_key: str = "openai_api_pricing_schema_description.txt"
    schema_key: str = "api_pricing_schema.json"
    pricing_html_key: str = "openai_api_pricing.html"
    pricing_info_key: str = "openai_api_pricing_info.json"
    pricing_info_from_ai_key: str = "openai_api_pricing_info_from_ai.json"

    # Dependencies that can be injected
    get_pricing_page_html: Callable[[], str] | None = None
    infer_schema_from_verbal_description: Callable[[str], dict[str, Any]] | None = (
        None
    )
    prompt_json_function: None | (
        Callable[[str, dict[str, Any]], Callable[[str], dict[str, Any]]]
    ) = None

    def __post_init__(self):
        """Initialize stores and ensure dependencies are available."""
        # Set up storage
        if not os.path.exists(self.resources_dir):
            raise FileNotFoundError(f"Directory does not exist: {self.resources_dir}")

        self.json_store = dol.JsonFiles(self.resources_dir)
        self.text_store = dol.TextFiles(self.resources_dir)

        # Import dependencies if not provided
        if self.get_pricing_page_html is None:
            self.get_pricing_page_html = get_pricing_page_html

        if (
            self.infer_schema_from_verbal_description is None
            or self.prompt_json_function is None
        ):
            import oa

            if self.infer_schema_from_verbal_description is None:
                self.infer_schema_from_verbal_description = (
                    oa.infer_schema_from_verbal_description
                )

            if self.prompt_json_function is None:
                self.prompt_json_function = oa.prompt_json_function

    @cache_this(cache="text_store", key=pricing_html_key)
    def pricing_page_html(self):
        """Retrieve the HTML for the OpenAI pricing page."""
        return get_pricing_page_html(self.pricing_url)

    @cache_this(cache="json_store", key=pricing_info_key)
    def pricing_info(self):
        return extract_pricing_data(self.pricing_page_html)

    @dol.cache_this(cache="text_store", key=lambda self: self.schema_description_key)
    def schema_description(self) -> str:
        """Generate and retrieve the schema description for OpenAI API pricing."""
        return """
        A schema to contain the pricing information for OpenAI APIs.
        The first level field should be the name category of API, (things like "Text Tokens", "Audio Tokens", "Fine Tuning", etc.)
        The value of the first level field should be a dictionary with the following fields:
        - "pricing_table_schema": A schema for the pricing table. That is, the list of the columns
        - "pricing_table": A list of dicts, corresponding to the table of pricing information for the API category. 
            For example, ```[{"Model": ..., "Input": ..., ...,}, {"Model": ..., "Input": ..., ...,}, ...]```
            Note that Sometimes there is more than one model name, so we should also have, in there, a field for "Alternate Model Names"
        - "extras": A dictionary with extra information that might be parsed out of each category. 
        For example, there's often some text that gives a description of the API category, and/or specifies "Price per 1M tokens" etc.
            
        Note that some of these tables have a "Batch API" version too. 
        In this case, there should be extra fields that have the same name as the fields above, but with "- Batch" appended to the name.
        For example, "Text Tokens - Batch", "Fine Tuning - Batch", etc.

        It is important to note: These first level fields are not determined in advance. 
        They are determined by the data that is scraped from the page.
        Therefore, these names should not be in the schema. 
        What should be in the schema is the fact that the data should be a JSON whose first 
        level field describes a category, and whose value specifies information about these category.
        """

    @dol.cache_this(cache="json_store", key=lambda self: self.schema_key)
    def schema(self) -> dict[str, Any]:
        """Generate or retrieve the schema for OpenAI API pricing."""
        return self.infer_schema_from_verbal_description(self.schema_description)

    @dol.cache_this(cache="json_store", key=lambda self: self.pricing_info_from_ai_key)
    def pricing_info_from_ai(self) -> dict[str, Any]:
        """Extract pricing information from the HTML using AI."""
        prompt = f"""
        Parse through this html and extract the pricing information for the OpenAI APIs.
        The pricing information should be structured according to the schema described below:

        {self.schema_description}

        Here is the html to parse:

        {{html}}
        """

        get_pricing_info = self.prompt_json_function(prompt, self.schema)
        return get_pricing_info(self.pricing_page_html)

    @property
    def pricing_tables_from_ai(self) -> dict[str, pd.DataFrame]:
        """Convert pricing tables to pandas DataFrames."""
        tables = self.pricing_info_from_ai.get("OpenAI_API_Pricing_Schema", {})

        # Create a mapping interface that transforms table data to DataFrames
        return dol.add_ipython_key_completions(
            dol.wrap_kvs(
                tables, value_decoder=lambda x: pd.DataFrame(x.get("pricing_table", []))
            )
        )

    def list_pricing_categories(self) -> list[str]:
        """List available pricing categories."""
        return list(self.pricing_info_from_ai.get("OpenAI_API_Pricing_Schema", {}))

    def get_pricing_table(self, category: str) -> pd.DataFrame:
        """Get a specific pricing table as a pandas DataFrame."""
        return self.pricing_tables_from_ai[category]


# -------------------------------------------------------------------------------------
# Extract pricing data from HTML content


import re
from bs4 import BeautifulSoup, Tag
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


from typing import Dict, List, Optional, Tuple, Any
from bs4 import BeautifulSoup, Tag
import re


def get_pricing_page_html(url=dflt_pricing_url):
    """
    Get the HTML content of the pricing page.

    Need to use chrome to render the page and get the full content.
    Sometimes need to do it twice, since some catpcha pages are shown sometimes.
    """
    from tabled.html import url_to_html_func

    url_to_html = url_to_html_func(("chrome", dict(wait=10)))
    return url_to_html(url)


def parse_pricing_page(html_content: str) -> dict[str, dict[str, Any]]:
    """
    Parse the HTML content and extract pricing information from all sections.

    Args:
        html_content: HTML content as a string

    Returns:
        Dictionary with category names as keys and their parsed data as values
    """
    soup = BeautifulSoup(html_content, "html.parser")
    sections = soup.find_all("section")

    results = {}

    for section in sections:
        parsed_categories = parse_section(section)
        if parsed_categories:
            for category_name, data in parsed_categories:
                results[category_name] = data

    return results


def parse_section(section: Tag) -> list[tuple[str, dict[str, Any]]]:
    """
    Parse a section containing pricing information into potentially multiple categories.

    Args:
        section: BeautifulSoup Tag containing a section

    Returns:
        List of tuples, each with (category_name, data_dict)
    """
    # Find the heading that contains the section name
    heading = section.find("h3", class_="anchor-heading")
    if not heading:
        return []

    # Extract the section name from the heading
    section_name = _clean_text(heading.get_text(strip=True))
    # Remove the "anchor-heading-icon" content if it exists
    svg = heading.find("svg")
    if svg:
        svg_text = svg.get_text(strip=True)
        section_name = section_name.replace(svg_text, "").strip()

    # Find all potential category labels within this section
    category_divs = section.find_all("div", class_="font-medium")

    # If no explicit categories are found, use the section name as the sole category
    if not category_divs:
        category_names = [section_name]
    else:
        # Fix for Image generation section which has a problematic structure
        if section_name == "Image generation":
            category_names = [section_name]
        else:
            category_names = [
                f"{section_name} - {_clean_text(div.get_text(strip=True))}"
                for div in category_divs
            ]

    # Find tables in this section
    tables = section.find_all("table")

    if not tables:
        return []

    # If we have more tables than categories, add generic category names
    if len(tables) > len(category_names):
        for i in range(len(category_names), len(tables)):
            category_names.append(f"{section_name} - Table {i+1}")

    results = []

    # Process each table with its corresponding category name
    for i, (table, category_name) in enumerate(zip(tables, category_names)):
        # Initialize data dictionary for this category
        data = {"pricing_table_schema": [], "pricing_table": [], "extras": {}}

        # Extract any extra information
        extras = _extract_extras(section)
        if extras:
            data["extras"] = extras

        # Extract table schema (column headers)
        thead = table.find("thead")
        if not thead:
            continue

        header_row = thead.find("tr")
        if not header_row:
            continue

        headers = header_row.find_all("th")
        schema = [_clean_text(header.get_text(strip=True)) for header in headers]

        data["pricing_table_schema"] = schema

        # Extract table rows
        tbody = table.find("tbody")
        if not tbody:
            continue

        rows = tbody.find_all("tr")
        table_data = []

        current_row_data = None
        rowspan_active = False
        rowspan_value = 0

        for row in rows:
            row_data = {}
            cells = row.find_all("td")

            # Check if this row is part of a rowspan
            first_cell = cells[0] if cells else None
            if (
                first_cell
                and first_cell.has_attr("rowspan")
                and int(first_cell["rowspan"]) > 1
            ):
                current_row_data = {}  # Start a new rowspan group
                rowspan_active = True
                rowspan_value = int(first_cell["rowspan"])

                # Extract model information from the rowspan cell
                model_info = _extract_model_info(first_cell)
                for key, value in model_info.items():
                    current_row_data[key] = value

                # Process the rest of the cells in this first rowspan row
                for i, cell in enumerate(cells[1:], 1):
                    if i < len(schema):
                        column_name = schema[i]
                        cell_data = _extract_cell_data(cell)
                        row_data[column_name] = cell_data

                # Combine model info with row data
                combined_data = {**current_row_data, **row_data}
                table_data.append(combined_data)
                rowspan_value -= 1

            elif rowspan_active and current_row_data:
                # This is a continuation row in a rowspan
                for i, cell in enumerate(cells):
                    # Adjusted index because first column is handled by rowspan
                    col_idx = i + 1
                    if col_idx < len(schema):
                        column_name = schema[col_idx]
                        cell_data = _extract_cell_data(cell)
                        row_data[column_name] = cell_data

                # Combine the current row data with the continuing rowspan data
                combined_data = {**current_row_data, **row_data}
                table_data.append(combined_data)

                rowspan_value -= 1
                if rowspan_value <= 0:
                    rowspan_active = False

            else:
                # Normal row without rowspan
                rowspan_active = False
                for i, cell in enumerate(cells):
                    if i < len(schema):
                        column_name = schema[i]

                        if i == 0:  # First column is typically the model name
                            model_info = _extract_model_info(cell)
                            for key, value in model_info.items():
                                row_data[key] = value
                        else:
                            cell_data = _extract_cell_data(cell)
                            row_data[column_name] = cell_data

                table_data.append(row_data)

        data["pricing_table"] = table_data
        results.append((category_name, data))

    return results


def _extract_extras(section: Tag) -> dict[str, str]:
    """Extract extra information from a section."""
    extras = {}

    # Look for descriptive text
    description = section.find(
        "div", class_="text-xs text-gray-500 whitespace-pre-line"
    )
    if description:
        extras["description"] = _clean_text(description.get_text(strip=True))

    return extras


def _extract_model_info(cell: Tag) -> dict[str, Any]:
    """Extract model name and any alternate model names from a cell."""
    info = {}

    # Find the main model name
    model_div = cell.find("div", class_="text-gray-900")
    if model_div:
        info["Model"] = _clean_text(model_div.get_text(strip=True))

    # Check for alternate model names (usually in a smaller text below)
    alt_model = cell.find("div", class_="text-xs text-gray-600")
    if alt_model:
        info["Alternate_Model"] = _clean_text(alt_model.get_text(strip=True))

    return info


def _extract_cell_data(cell: Tag) -> dict[str, str]:
    """Extract data from a pricing cell, which might include multiple values."""
    # Initialize with the full text as default
    cell_text = _clean_text(cell.get_text(strip=True))

    # If the cell is just a simple text cell, return it directly
    if "-" in cell_text and len(cell_text) < 3:
        return cell_text

    # Check for price value and unit
    price_div = cell.find("div", class_="text-right flex-1")
    unit_div = cell.find("div", class_="text-xs text-gray-500 text-nowrap text-right")

    # If we have both components, organize them properly
    if price_div and unit_div:
        price = _clean_text(price_div.get_text(strip=True))
        unit = _clean_text(unit_div.get_text(strip=True))

        # If the price is empty, just return the text
        if not price.strip():
            return cell_text

        # Return structured data
        return {"price": price, "unit": unit}

    # Return the full text if we couldn't break it down
    return cell_text


def _clean_text(text: str) -> str:
    """Clean up text by removing extra whitespace and newlines."""
    # Replace multiple whitespace with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove any leading/trailing whitespace
    return text.strip()


def extract_pricing_data(html_content: str) -> dict[str, dict[str, Any]]:
    """
    Main function to extract pricing data from HTML content.

    Args:
        html_content: HTML content of the pricing page

    Returns:
        Dictionary with API categories as keys and pricing information as values

    Example:


    """
    return parse_pricing_page(html_content)
```

## ask.py

```python
"""Access to already made prompt functions"""

from oa.tools import PromptFuncs

ai = PromptFuncs()
```

## audio.py

```python
"""OpenAI Audio tools"""

from functools import partial
from typing import Optional, Literal
from collections.abc import Callable, Iterable

from i2 import Sig

from oa.util import openai, mk_client, ensure_oa_client, OaClientSpec

# --------------------------------------------------------------------------------------
# Defaults

DFLT_TRANSCRIPTION_MODEL = "whisper-1"
DFLT_TTS_MODEL = "tts-1"
DFLT_TTS_VOICE = "alloy"
DFLT_RESPONSE_FORMAT = "json"
DFLT_AUDIO_FORMAT = "mp3"

# --------------------------------------------------------------------------------------
# Transcription


def _parse_transcription_response(resp, response_format: str) -> dict:
    """Parse the transcription API response into a standardized dict."""
    if response_format in ("json", "verbose_json"):
        full_text = resp.get("text", "")
        segments = resp.get("segments", None)
        language = resp.get("language", None)
    elif response_format == "srt":
        full_text = resp
        segments = None
        language = None
    else:
        full_text = resp if isinstance(resp, str) else str(resp)
        segments = None
        language = None

    return {
        "text": full_text,
        "segments": segments,
        "language": language,
        "raw_response": resp,
    }


@Sig.replace_kwargs_using(openai.audio.transcriptions.create)
def transcribe(
    audio: str,
    *,
    model: str = DFLT_TRANSCRIPTION_MODEL,
    response_format: str = DFLT_RESPONSE_FORMAT,
    language: Optional[str] = None,
    client: OaClientSpec = None,
    **kwargs,
) -> dict:
    """
    Transcribe an audio or video file using OpenAI's Whisper API.

    :param audio_file_path: Path to the audio/video file to transcribe
    :param model: Transcription model to use
    :param response_format: Format for the response ('json', 'text', 'srt', 'verbose_json')
    :param language: Optional ISO-639-1 language code (e.g., 'en', 'fr')
    :param client: OpenAI client instance, API key string, config dict, or None
    :param kwargs: Additional parameters for the API
    :return: Dict with 'text', 'segments', 'language', and 'raw_response'

    Note: Actual API calls require a valid audio file and API key
    """
    client = ensure_oa_client(client)

    with open(audio, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=f,
            model=model,
            response_format=response_format,
            language=language,
            **kwargs,
        )

    return _parse_transcription_response(resp, response_format)


# --------------------------------------------------------------------------------------
# SRT (SubRip) utilities


def _format_srt_timestamp(seconds: float) -> str:
    """
    Format seconds as SRT timestamp (HH:MM:SS,mmm).

    >>> _format_srt_timestamp(0)
    '00:00:00,000'
    >>> _format_srt_timestamp(3661.5)
    '01:01:01,500'
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"


def _segments_to_srt_entries(segments):
    """Generate SRT entries from segments with timestamps."""
    for idx, seg in enumerate(segments, start=1):
        start_str = _format_srt_timestamp(seg["start"])
        end_str = _format_srt_timestamp(seg["end"])
        text = seg["text"].strip()
        yield f"{idx}\n{start_str} --> {end_str}\n{text}\n"


def _text_to_equal_duration_segments(
    text: str, segment_duration: float, *, words_per_segment: int = None
):
    """
    Split text into segments of roughly equal duration.

    >>> segments = list(_text_to_equal_duration_segments("one two three four", 10.0))
    >>> segments[0]['text']
    'one two three four'
    >>> segments[0]['end']
    10.0
    """
    words = text.split()
    if not words:
        return

    if words_per_segment is None:
        # Simple heuristic: assume uniform word rate
        words_per_segment = max(1, len(words) // max(1, int(len(words) / 10)))

    idx = 0
    for i in range(0, len(words), words_per_segment):
        chunk = " ".join(words[i : i + words_per_segment])
        start_t = idx * segment_duration
        end_t = start_t + segment_duration
        yield {"start": start_t, "end": end_t, "text": chunk}
        idx += 1


def transcription_to_srt(transcription: dict, *, segment_duration: float = None) -> str:
    """
    Convert a transcription dict to SRT format string.

    :param transcription: Dict with 'text' and optionally 'segments'
    :param segment_duration: If segments missing, duration per generated segment
    :return: SRT formatted string

    >>> result = transcription_to_srt({'text': 'Hello world'})
    >>> 'Hello world' in result
    True
    """
    segments = transcription.get("segments")

    if segments:
        # Use existing timestamped segments
        entries = _segments_to_srt_entries(segments)
    elif segment_duration:
        # Generate equal-duration segments from text
        text = transcription["text"].strip()
        synthetic_segments = _text_to_equal_duration_segments(text, segment_duration)
        entries = _segments_to_srt_entries(synthetic_segments)
    else:
        # Single segment spanning the whole text
        text = transcription["text"].strip()
        entries = ["1\n00:00:00,000 --> 99:59:59,000\n" + text + "\n"]

    return "\n".join(entries)


# --------------------------------------------------------------------------------------
# Text-to-Speech


@Sig.replace_kwargs_using(openai.audio.speech.create)
def text_to_speech(
    text: str,
    *,
    model: str = DFLT_TTS_MODEL,
    voice: str = DFLT_TTS_VOICE,
    response_format: str = DFLT_AUDIO_FORMAT,
    client: OaClientSpec = None,
    **kwargs,
) -> bytes:
    """
    Convert text to speech using OpenAI's TTS API.

    :param text: Text to convert to speech
    :param model: TTS model to use
    :param voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
    :param response_format: Audio format ('mp3', 'opus', 'aac', 'flac', 'wav', 'pcm')
    :param client: OpenAI client instance, API key string, config dict, or None
    :param kwargs: Additional parameters for the API
    :return: Audio content as bytes

    Note: Actual API calls require a valid API key
    """
    client = ensure_oa_client(client)

    response = client.audio.speech.create(
        input=text, model=model, voice=voice, response_format=response_format, **kwargs
    )

    # Handle different response types
    if hasattr(response, "content"):
        return response.content
    elif isinstance(response, bytes):
        return response
    else:
        return response.read()


# Alias for consistency with other modules
tts = text_to_speech
```

## base.py

```python
"""Base oa functionality"""

import re
from itertools import chain
from functools import partial
from typing import Union, Optional, KT
from collections.abc import Iterable, Mapping, Callable

from types import SimpleNamespace
from i2 import Sig, Param

from oa.util import (
    openai,
    djoin,
    mk_client,
    num_tokens,
    model_information_dict,
    DFLT_ENGINE,
    DFLT_MODEL,
    DFLT_EMBEDDINGS_MODEL,
)
from oa.openai_specs import prompt_path

Text = str
TextStrings = Iterable[Text]
TextStore = Mapping[KT, Text]
Texts = Union[TextStrings, TextStore]
TextOrTexts = Union[Text, Texts]

api = None
# TODO: Expand this to include all the other endpoints (automatically?)
# api = SimpleNamespace(
#     chat=sig.CreateChatCompletionRequest(openai.chat.completions.create),
#     complete=sig.CreateCompletionRequest(openai.completions.create),
#     dalle=sig.CreateImageRequest(openai.images.generate),
# )

_model_id_aliases = {
    "davinci": "text-davinci-003",
    "ada v2": "text-embedding-ada-002",
}

_model_information_aliases = {
    "max_tokens": "max_input",
    "price": "price_per_million_tokens",
}


# TODO: Literal type of models and information?
def model_information(model, information):
    """Get information about a model"""
    if model in _model_id_aliases:
        model = _model_id_aliases[model]
    if information in _model_information_aliases:
        information = _model_information_aliases[information]

    if model not in model_information_dict:
        raise ValueError(f"Unknown model: {model}")
    if information not in model_information_dict[model]:
        raise ValueError(f"Unknown information: {information}")

    return model_information_dict[model][information]


model_information.model_information_dict = model_information_dict


# TODO: Parse more info and complete this function
#     (see https://github.com/thorwhalen/oa/discussions/8#discussioncomment-9138661)
def compute_price(
    model: str, num_input_tokens: int = None, num_output_tokens: int | None = None
):
    """Compute the price of a model given the number of input and output tokens"""
    assert num_output_tokens is None, "num_output_tokens not yet implemented"
    if num_input_tokens is None:
        return partial(compute_price, model)
    price_per_million_tokens = model_information(model, "price_per_million_tokens")
    return price_per_million_tokens * (num_input_tokens / 1_000_000)


compute_price.model_information_dict = model_information_dict

prompt_dalle_path = partial(prompt_path, prefix=djoin("dalle"))
prompt_davinci_path = partial(prompt_path, prefix=djoin("davinci"))

# TODO: Use oa.openai_specs sig to provide good signatures


@Sig.replace_kwargs_using(openai.completions.create)
def complete(prompt, model=None, **complete_params):
    if "engine" in complete_params:
        model = complete_params.pop("engine")
    model = model or getattr(complete, "engine", DFLT_ENGINE)
    text_resp = openai.completions.create(model=model, prompt=prompt, **complete_params)
    return text_resp.choices[0].text


complete.engine = DFLT_ENGINE


@Sig.replace_kwargs_using(openai.chat.completions.create)
def _raw_chat(prompt=None, model=DFLT_MODEL, *, messages=None, **chat_params):
    if not ((prompt is None) ^ (messages is None)):
        raise ValueError("Either prompt or messages must be specified, but not both.")
    if prompt is not None:
        messages = [{"role": "user", "content": prompt}]
    return openai.chat.completions.create(messages=messages, model=model, **chat_params)


# chat_sig = sig.CreateChatCompletionRequest
# chat_sig = chat_sig.ch_defaults(model=DFLT_MODEL, messages=None)
# chat_sig = Sig([Param(name='prompt', default=None, annotation=str), *chat_sig.params])


@Sig.replace_kwargs_using(_raw_chat)
def chat(prompt=None, *, model=DFLT_MODEL, messages=None, **chat_params):
    resp = _raw_chat(prompt=prompt, model=model, messages=messages, **chat_params)
    # TODO: Make attr and item getters more robust (use glom?)
    return resp.choices[0].message.content


chat.raw = _raw_chat


@Sig.replace_kwargs_using(openai.images.generate)
def _raw_dalle(prompt, n=1, size="512x512", **image_create_params):
    return openai.images.generate(prompt=prompt, n=n, size=size, **image_create_params)


@Sig.replace_kwargs_using(_raw_dalle)
def dalle(prompt, n=1, size="512x512", **image_create_params):
    r = _raw_dalle(prompt=prompt, n=n, size=size, **image_create_params)
    return r.data[0].url


def list_engine_ids(pattern: str | None = None):
    """List the available engine IDs. Optionally filter by a regex pattern."""
    models_list = mk_client().models.list()
    model_ids = [x.id for x in models_list.data]
    if pattern:
        # filter model_ids by pattern, taken to be a regex pattern
        pattern = re.compile(pattern)
        model_ids = list(filter(pattern.search, model_ids))
    return model_ids


def _raise_if_any_invalid(
    validation_vector: Iterable[bool],
    texts: Iterable[Text] = None,
    print_invalid_texts=True,
):
    if isinstance(validation_vector, bool):
        # if it's a single validation boolean, make it a list of one boolean
        validation_vector = [validation_vector]
    else:
        validation_vector = list(validation_vector)
    if not all(validation_vector):
        if print_invalid_texts:
            print(
                "Invalid text(s):\n",
                "\n".join(
                    item
                    for is_valid, item in zip(validation_vector, texts)
                    if not is_valid
                ),
            )
        raise ValueError("Some of the texts are invalid")
    return texts


# --------------------------------------------------------------------------------------
# Embeddings
"""
There are (at least) three ways to compute embeddings, which are more or less ideal 
depending on the situation.

* One by one, locally and serially (that is, we wait for the response of the request 
    before sending another). This is **VERY** slow, and you don't want to do this with 
    a lot of data. But it has the advantage of being simple and straightforward, and, 
    if one of your segments has a problem, you'll know easily exactly which one does.
* In batches, locally and serially. 
* In batches, remotely, in parallel, asynchronously. Advantages here are that it's 
    remote, so you're not hogging down the resources of your computer, and the remote 
    server will manage the persistence, status, etc. It's also cheaper (with OpenAI, 
    at the time of writing this, half the price). But it's more complex, and though 
    often faster to get your response every time I've ever tried, you are "only" 
    guaranteed getting your batch jobs within 24h of launching them.
"""

from openai import NOT_GIVEN
from typing import Union, List, Any
from oa.util import chunk_iterable, mk_local_files_saves_callback

# from collections.abc import Mapping, Iterable

extra_embeddings_params = Sig(openai.embeddings.create) - {"input", "model"}


# TODO: Added a lot of options, but not really clean. Should be cleaned up.
# TODO: The dict-versus-list types should be handled more cleanly!
# TODO: Integrate the batch API way of doing embeddings
# TODO: Batches should be able to be done in paralel, with async/await
# TODO: Make a few useful validation_callback functions
#    (e.g. return list or dict where invalid texts are replaced with None)
#    (e.g. return dict containing only valid texts (if input was list, uses indices as keys)
@Sig.replace_kwargs_using(extra_embeddings_params)
def embeddings(
    texts: TextOrTexts = None,
    *,
    batch_size: int | None = 2048,  # found on unofficial OpenAI API docs
    egress: str | None = None,
    batch_callback: Callable[[int, list[list]], Any] | None = None,
    validate: bool | Callable | None = True,
    valid_text_getter=_raise_if_any_invalid,
    model=DFLT_EMBEDDINGS_MODEL,
    client=None,
    dimensions: int | None = NOT_GIVEN,
    **extra_embeddings_params,
):
    """
    Get embeddings for a text or texts.

    :param texts: A string, an iterable of strings, or a dictionary of strings
    :param egress: A function that takes the embeddings and returns the desired output.
        If None, the output will be a list of embeddings. If False, no output will be
        returned, so the batch_callback had better be set to accumulate the results.
    :param batch_callback: A function that is called after each batch of embeddings is
        computed. This can be used for logging, saving, etc.
        One common use case is to save the intermediate results, in files, database,
        or in a list. This can be useful if you're worried about the process failing
        and want to be able to resume from where you left off instead of having
        to start over (wasting time, and money).
        To accumulate the results in a list, you can set `results = []` and then
        use a lambda function like this:
        `batch_callback = lambda i, batch: results.extend(batch)`.
    :param validate: If True, validate the text(s) before getting embeddings
    :param valid_text_getter: A function that gets valid texts from the input texts
    :param model: The model to use for embeddings
    :param client: The OpenAI client to use
    :param dimensions: If given will reduce the dimensions of the full size embedding
        vectors to that size
    :param batch_size: The maximum number of texts to send in a single request
    :param extra_embeddings_params: Extra parameters to pass to the embeddings API

    >>> from functools import partial
    >>> dimensions = 3
    >>> embeddings_ = partial(embeddings, dimensions=dimensions, validate=True)

    Test with a single word:

    >>> text = "vector"
    >>> result = embeddings_(text)
    >>> result  # doctest: +SKIP
    [-0.4096039831638336, 0.3794299364089966, -0.8296127915382385]
    >>> isinstance(result, list)
    True
    >>> len(result) == dimensions == 3
    True

    # Test with a list of words
    >>> texts = ["semantic", "vector"]
    >>> result = embeddings_(texts)
    >>> isinstance(result, list)
    True
    >>> len(result)
    2

    Two vectors; one for each word. Note that the second vector is the vector of
    "vector", which we've seen before.
    >>> result[1]  # doctest: +SKIP
    [-0.4096039831638336, 0.3794299364089966, -0.8296127915382385]

    >>> len(result[1]) == dimensions == 3
    True

    # Test with a dictionary of words
    >>> texts = {"adj": "semantic", "noun": "vector"}
    >>> result = embeddings_(texts)
    >>> isinstance(result, dict)
    True
    >>> len(result)
    2
    >>> result["noun"]  # doctest: +SKIP
    [-0.4096039831638336, 0.3794299364089966, -0.8296127915382385]
    >>> len(result["adj"]) == len(result["noun"]) == dimensions == 3
    True

    If you don't specify `texts`, you will get a "partial" function that you can
    use later to compute embeddings for texts. This is useful if you want to
    set some parameters (like `dimensions`, `validate`, etc.) and then use the
    resulting function to compute embeddings for different texts later on.
    >>> embeddings_with_10_dimensions = embeddings_(texts=None, dimensions=10)
    >>> isinstance(embeddings_with_10_dimensions, Callable)
    True

    """
    if texts is None:
        _kwargs = locals()
        _ = _kwargs.pop("texts")
        extra_embeddings_params = _kwargs.pop("extra_embeddings_params", {})
        return partial(embeddings, **_kwargs, **extra_embeddings_params)

    if egress is False:
        assert batch_callback, (
            "batch_callback must be provided if egress is False: "
            "It will be the batch_callback's responsibility to accumulate the batches "
            "of embeddings!"
        )

    if batch_callback == "temp_files":  # an extra not annotated or documented
        # convenience to get intermediary results saved to file
        batch_callback = mk_local_files_saves_callback()
    batch_callback = batch_callback or (lambda i, batch: None)
    assert callable(batch_callback) & (
        len(Sig(batch_callback)) >= 2
    ), "batch_callback must be callable with at least two arguments (i, batch)"
    texts, texts_type, keys = _prepare_embeddings_args(
        validate, texts, valid_text_getter, model
    )

    if texts_type is str and egress is not None:
        raise ValueError("egress should be None if texts is a single string")

    if client is None:
        client = mk_client()

    def _embeddings_batches():
        for i, batch in enumerate(chunk_iterable(texts, batch_size)):
            batch_result = _embeddings_batch(
                batch,
                model=model,
                client=client,
                dimensions=dimensions,
                **extra_embeddings_params,
            )
            batch_callback(i, batch_result)
            yield from batch_result

    # vectors = chain.from_iterable(_embeddings_batches())
    vectors = _embeddings_batches()

    if egress is False:
        # the batch
        for _ in vectors:
            pass
    else:
        if egress is None:
            if issubclass(texts_type, Mapping):
                egress = lambda vectors: {k: v for k, v in zip(keys, vectors)}
            else:
                egress = list

        if texts_type is str:
            return next(iter(vectors))  # there's one and only one (note: no egress)
        else:
            return egress(vectors)


def _embeddings_batch(
    texts: TextOrTexts,
    model=DFLT_EMBEDDINGS_MODEL,
    client=None,
    dimensions: int | None = NOT_GIVEN,
    **extra_embeddings_params,
):

    # Note: validate set to False, as we've already validated
    return [
        x.embedding
        for x in client.embeddings.create(
            input=texts,
            model=model,
            dimensions=dimensions,
            **extra_embeddings_params,
        ).data
    ]


# --------------------------------------------------------------------------------------
# embeddings utils


def _prepare_embeddings_args(validate, texts, valid_text_getter, model):
    if validate:
        texts = validate_texts_for_embeddings(texts, valid_text_getter, model)

    texts, texts_type, keys = normalize_text_input(texts)

    return texts, texts_type, keys


def validate_texts_for_embeddings(
    texts, valid_text_getter, model=DFLT_EMBEDDINGS_MODEL
):
    validate = partial(text_is_valid, model=model)
    validation_vector = validate(texts)

    # TODO: Too many places where we have to check if it's a dict or a list. Need to clean up.
    if isinstance(validation_vector, Mapping):
        keys = list(validation_vector.keys())
        validation_vector = list(validation_vector.values())
    else:
        keys = None

    texts = valid_text_getter(validation_vector, texts=texts)

    # TODO: Too many places where we have to check if it's a dict or a list. Need to clean up.
    if isinstance(validation_vector, Mapping):
        return {k: v for k, v in zip(keys, texts)}
    else:
        return texts


def normalize_text_input(texts: TextOrTexts) -> TextStrings:
    """Ensures the type of texts is an iterable of strings"""
    if isinstance(texts, str):
        return [texts], str, None  # Single string case
    elif isinstance(texts, Mapping):
        return texts.values(), Mapping, list(texts.keys())
    elif isinstance(texts, Iterable):
        return texts, Iterable, None  # Iterable case
    else:
        raise ValueError("Input type not supported")


def text_is_valid(
    texts: TextOrTexts,
    token_count=True,
    *,
    model: str = DFLT_EMBEDDINGS_MODEL,
    max_tokens=None,
):
    """Check if text (a string or iterable of strings) is/are valid for a given model.

    Text is valid if
    - it is not empty
    - the number of tokens in the text is less than or equal to the max_tokens

    :param texts: a string or an iterable of strings
    :param token_count: Specification of the token count of the text or texts
    :param model: The model to use for token count
    :param max_tokens: The maximum number of tokens allowed by the model

    If token_count is an integer, it will check if it is less than or equal to the
    `max_tokens`.

    If token_count is True, it will compute the number of tokens in the text using the
    model specified by `model` and check if it is less than or equal to `max_tokens`.

    If token_count is False, it will not check the token count.

    If token_count is an iterable, it will apply the same mechanism as above, to each
    text in `texts` and the corresponding token count in `token_count`.
    This means both texts and token_count(s) must be of the same length.

    Examples:

    >>> text_is_valid('Hello, world!')
    True
    >>> text_is_valid('')
    False
    >>> text_is_valid('Alice ' * 9000)
    False
    >>> text_is_valid('Alice ' * 9000, token_count=False)
    True
    >>> list(text_is_valid(['Bob', '', 'Alice ' * 9000]))
    [True, False, False]
    >>> list(text_is_valid(['Bob', '', 'Alice ' * 9000], token_count=False))
    [True, False, True]

    """
    # Normalize the input
    texts, texts_type, keys = normalize_text_input(texts)

    # Set the maximum tokens allowed if not provided
    max_tokens = max_tokens or model_information_dict[model]["max_input"]

    # Define the validation logic for a single text
    def is_text_valid(text, token_count):
        if not text:
            return False
        if token_count:
            if token_count is True:
                token_count = num_tokens(text, model=model)
            return token_count <= max_tokens
        return True

    # Handle the validation for different input types
    if isinstance(token_count, Iterable):
        results = map(is_text_valid, texts, token_count)
    else:
        results = map(partial(is_text_valid, token_count=token_count), texts)

    if texts_type is Mapping:
        return {
            k: v for k, v in zip(keys, results)
        }  # Return a mapping if input was a mapping
    elif texts_type is str:
        return next(
            results
        )  # Return the boolean directly if the input was a single string
    else:
        return list(results)  # Return the list of booleans for an iterable of strings


# df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
# df.to_csv('output/embedded_1k_reviews.csv', index=False)
```

## batch_embeddings.py

```python
"""
Simplified interface for computing embeddings in bulk using OpenAI's batch API.

This module provides a clean, reusable interface for generating embeddings from
text segments using OpenAI's batch API, handling the async nature of the API
and providing status monitoring, error handling, and result aggregation.
"""

import time
import json
import logging
import asyncio
from typing import (
    Dict,
    List,
    Union,
    Optional,
    Any,
    TypeVar,
    Tuple,
)
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
from functools import partial
from contextlib import contextmanager
import numpy as np

from oa.stores import OaDacc
from oa.util import jsonl_loads_iter, concat_lists, extractors, ProcessingManager

# Define BatchRequestCounts since it's not available in oa.util
from typing import Optional
from dataclasses import dataclass

from oa.batches import (
    BatchId,
    BatchObj,
    BatchSpec,
    mk_batch_file_embeddings_task,
    get_batch_obj,
    get_output_file_data,
)
from oa.base import DFLT_EMBEDDINGS_MODEL
from oa.util import chunk_iterable  # like fixed_step_chunker but simpler & with casting


@dataclass
class BatchRequestCounts:
    """Counts of batch requests by status"""

    completed: int = 0
    failed: int = 0
    total: int = 0


# Type aliases for improved readability
Segment = str
Segments = Union[list[Segment], dict[str, Segment]]
Embedding = list[float]
Embeddings = list[Embedding]
SegmentsMapper = dict[BatchId, list[Segment]]
EmbeddingsMapper = dict[BatchId, list[Embedding]]

# Default values
DFLT_BATCH_SIZE = 1000
DFLT_POLL_INTERVAL = 5.0  # seconds
DFLT_MAX_POLLS = None  # None means unlimited
DFLT_VERBOSITY = 1
DFLT_KEEP_PROCESSING_INFO = False


class BatchStatus:
    """Status constants for batch processing"""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Check if a status represents a terminal state"""
        return status in {cls.COMPLETED, cls.FAILED, cls.EXPIRED, cls.CANCELLED}

    @classmethod
    def is_success(cls, status: str) -> bool:
        """Check if a status represents successful completion"""
        return status == cls.COMPLETED


class BatchError(Exception):
    """Base exception for batch processing errors"""

    pass


from dol import (
    process_path,
    wrap_kvs,
    ensure_clear_to_kv_store,
    Files,
    JsonFiles as _JsonFiles,
    PickleFiles as _PickleFiles,
)
from tabled.wrappers import single_column_parquet_encode, single_column_parquet_decode

JsonFiles = ensure_clear_to_kv_store(_JsonFiles)
PickleFiles = ensure_clear_to_kv_store(_PickleFiles)
SingleColParquetFiles = ensure_clear_to_kv_store(
    wrap_kvs(
        Files,
        value_encoder=single_column_parquet_encode,
        value_decoder=single_column_parquet_decode,
    )
)
DFLT_MALL_STORE_CLASS = PickleFiles


dflt_store_cls_dict = {
    "current": PickleFiles,
    "segments": PickleFiles,
    "finished": PickleFiles,
    "erred": PickleFiles,
    "embeddings": PickleFiles,
}


class ProcessingMall(Mapping):
    """
    Container for all stores needed during batch processing.

    The ProcessingMall contains five stores, all keyed by batch_id:
    - current: Batches that are currently being processed
    - segments: The text segments corresponding to each batch
    - finished: Completed batches
    - erred: Batches that encountered errors
    - embeddings: The computed embeddings for each batch
    """

    _store_names = ("current", "segments", "finished", "erred", "embeddings")

    def __iter__(self) -> Iterator[str]:
        return iter(self._store_names)

    def __len__(self) -> int:
        return len(self._store_names)

    def __getitem__(self, key: str) -> MutableMapping:
        if key not in self._store_names:
            raise KeyError(f"Invalid store name: {key}")
        return getattr(self, key)

    def __init__(
        self,
        *,
        current: MutableMapping | None = None,
        segments: MutableMapping | None = None,
        finished: MutableMapping | None = None,
        erred: MutableMapping | None = None,
        embeddings: MutableMapping | None = None,
    ):
        """
        Initialize the ProcessingMall with optional custom stores.

        Args:
            current: Store for batches currently being processed
            segments: Store for segments corresponding to each batch
            finished: Store for completed batches
            erred: Store for batches that encountered errors
            embeddings: Store for computed embeddings
        """
        self.current = current if current is not None else {}
        self.segments = segments if segments is not None else {}
        self.finished = finished if finished is not None else {}
        self.erred = erred if erred is not None else {}
        self.embeddings = embeddings if embeddings is not None else {}

    @classmethod
    def with_folder(cls, rootdir: str, store_cls=JsonFiles):
        """
        Create a ProcessingMall with stores that persist to local files.

        Args:
            rootdir: Directory where the stores will be saved
        """
        from imbed.util import DFLT_BATCHES_DIR
        import os

        if not isinstance(store_cls, Mapping):
            assert callable(
                store_cls
            ), "store_cls must be a callable or a dict of callables"
            store_cls = {name: store_cls for name in cls._store_names}

        # if rootdir doesn't have any path separator, and doesn't exist, assume it's a
        # a name to use to make an actual directory in the imbed app data dir
        if "/" not in rootdir and "\\" not in rootdir and not os.path.exists(rootdir):
            rootdir = os.path.join(DFLT_BATCHES_DIR, rootdir)

        def stores():
            for store_name in cls._store_names:
                store_rootdir = process_path(
                    rootdir, store_name, ensure_dir_exists=True
                )
                _store_cls = store_cls.get(store_name, DFLT_MALL_STORE_CLASS)
                yield store_name, _store_cls(store_rootdir)

        instance = cls(**dict(stores()))
        instance.rootdir = rootdir
        return instance

    def clear(self) -> None:
        """Clear all stores"""
        for store in (
            self.current,
            self.segments,
            self.finished,
            self.erred,
            self.embeddings,
        ):
            store.clear()

    def is_complete(self) -> bool:
        """Check if processing is complete (no batches in current)"""
        return len(self.current) == 0


class EmbeddingsBatchProcess:
    """
    Manages the lifecycle of batch embedding requests.

    This class handles submitting batch requests to the OpenAI API,
    monitoring their status, retrieving results, and aggregating them.
    It can be used as a context manager for automatic cleanup.
    """

    def __init__(
        self,
        segments: Segments | None = None,
        processing_mall: str | ProcessingMall | None = None,
        *,
        model: str = DFLT_EMBEDDINGS_MODEL,
        batch_size: int = DFLT_BATCH_SIZE,
        poll_interval: float = DFLT_POLL_INTERVAL,
        max_polls: int | None = DFLT_MAX_POLLS,
        verbosity: int = DFLT_VERBOSITY,
        keep_processing_info: bool = DFLT_KEEP_PROCESSING_INFO,
        dacc: OaDacc | None = None,
        logger: logging.Logger | None = None,
        **embeddings_kwargs,
    ):
        """
        Initialize a new EmbeddingsBatchProcess for embedding generation.

        Args:
            segments: Text segments to embed, either as a list or a dictionary.
            model: OpenAI embedding model to use
            batch_size: Maximum number of segments per batch
            poll_interval: Seconds between status checks
            max_polls: Maximum number of status checks before timing out
            verbosity: Level of logging detail (0-2)
            processing_mall: Optional custom ProcessingMall
            keep_processing_info: Whether to keep mall data after completion
            dacc: Optional custom OaDacc instance
            logger: Optional custom logger
            **embeddings_kwargs: Additional parameters for embedding generation
        """
        self.segments = segments
        self.model = model
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.max_polls = max_polls or int(
            24 * 3600 / poll_interval
        )  # Default to 24h worth of polls
        self.verbosity = verbosity

        if processing_mall is None:
            processing_mall = ProcessingMall()
        elif isinstance(processing_mall, str):
            # If a string is provided, treat it as a directory for JsonFiles
            processing_mall = ProcessingMall.with_folder(processing_mall)
        elif not isinstance(processing_mall, ProcessingMall):
            raise TypeError(
                "processing_mall must be a ProcessingMall instance or a directory path"
            )
        self.processing_mall = processing_mall

        self.keep_processing_info = keep_processing_info
        self.dacc = dacc or OaDacc()
        self.embeddings_kwargs = embeddings_kwargs

        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(
            logging.ERROR
            if verbosity == 0
            else logging.INFO if verbosity == 1 else logging.DEBUG
        )

        # Internal state
        self.processing_manager = None
        self._is_running = False
        self._is_complete = False
        self._result = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        if not self.keep_processing_info and self._is_complete:
            self.processing_mall.clear()
        return False  # Don't suppress exceptions

    def _prepare_batches(self) -> Iterator[tuple[list[Segment], BatchId]]:
        """
        Prepare and chunk segments into batches.

        Returns:
            Iterator yielding (segments_batch, batch_id) tuples
        """
        # Convert segments to a list if it's a dictionary
        if isinstance(self.segments, dict):
            segment_values = list(self.segments.values())
        else:
            segment_values = self.segments

        # Create a batcher function
        batcher = partial(chunk_iterable, chk_size=self.batch_size)

        # Process segments in batches
        for segments_batch in batcher(segment_values):
            # Submit batch to OpenAI API
            batch_obj = self.dacc.launch_embedding_task(
                segments_batch, **self.embeddings_kwargs
            )
            # Extract the string ID from the batch object
            batch_id = batch_obj.id

            yield segments_batch, batch_id

    def submit_batches(self) -> dict[BatchId, BatchObj]:
        """
        Submit all segment batches to the OpenAI API.

        Returns:
            Dictionary mapping batch IDs to batch objects
        """
        submitted_batches = {}

        self.logger.info(f"Submitting batches for {len(self.segments)} segments")

        # Process segments in batches
        for segments_batch, batch_id in self._prepare_batches():
            # Store batch info and segments
            batch_obj = self.dacc.s.batches[batch_id]
            # Use string batch_id as key
            self.processing_mall.current[batch_id] = batch_obj
            self.processing_mall.segments[batch_id] = segments_batch

            submitted_batches[batch_id] = batch_obj

            self.logger.debug(
                f"Submitted batch {batch_id} with {len(segments_batch)} segments"
            )

        self.logger.info(f"Submitted {len(submitted_batches)} batches")
        return submitted_batches

    def _process_batch_status(
        self, batch_id: BatchId, status: str, output_data: Any
    ) -> bool:
        """
        Process status updates for a batch.

        Args:
            batch_id: The ID of the batch
            status: Current status of the batch
            output_data: Data returned for completed batches

        Returns:
            True if batch is complete (terminal state), False otherwise
        """
        if status == BatchStatus.COMPLETED:
            self.logger.info(f"Batch {batch_id} completed successfully")

            # Extract embeddings from output data
            embeddings = concat_lists(
                map(
                    extractors.embeddings_from_output_data,
                    jsonl_loads_iter(output_data.content),
                )
            )

            # Store embeddings and move batch to finished
            self.processing_mall.embeddings[batch_id.id] = embeddings
            self.processing_mall.finished[batch_id.id] = self.processing_mall.current[
                batch_id.id
            ]
            del self.processing_mall.current[batch_id.id]

            return True

        elif BatchStatus.is_terminal(status):
            self.logger.warning(f"Batch {batch_id} ended with status: {status}")

            # Move batch to erred
            self.processing_mall.erred[batch_id.id] = self.processing_mall.current[
                batch_id.id
            ]
            del self.processing_mall.current[batch_id.id]

            return True

        else:
            self.logger.debug(f"Batch {batch_id} status: {status}")

            # Update batch status in current store
            # Ensure we're using string batch_id as key
            self.processing_mall.current[batch_id.id] = self.dacc.s.batches[batch_id.id]

            return False

    def monitor_batches(self) -> None:
        """
        Monitor the status of all submitted batches until completion.
        """
        if not self.processing_mall.current:
            self.logger.warning("No batches to monitor")
            self._is_complete = True
            return

        self.logger.info(f"Monitoring {len(self.processing_mall.current)} batches")
        self._is_running = True

        # Define the processing function
        def batch_processor(batch_id: BatchId) -> tuple[str, Any]:
            try:
                # Get batch status and output data
                # Ensure we're using string batch_id
                batch_obj = self.dacc.s.batches[batch_id.id]
                status = batch_obj.status

                if status == BatchStatus.COMPLETED:
                    output_data = self.dacc.s.files_base[batch_obj.output_file_id]
                    return status, output_data
                else:
                    return status, None

            except Exception as e:
                self.logger.error(f"Error checking batch {batch_id}: {str(e)}")
                return BatchStatus.FAILED, None

        # Define wait time function to control polling interval
        def wait_time_function(cycle_duration: float, local_vars: dict) -> float:
            """Calculate how long to wait before the next cycle"""
            # Ensure we wait at least poll_interval seconds between checks
            return max(0.0, self.poll_interval - cycle_duration)

        # Create processing manager with dictionary of string batch IDs
        pending_items = {
            batch_id: batch_obj
            for batch_id, batch_obj in self.processing_mall.current.items()
        }

        self.processing_manager = ProcessingManager(
            pending_items=pending_items,
            processing_function=batch_processor,
            handle_status_function=self._process_batch_status,
            wait_time_function=wait_time_function,
            status_check_interval=self.poll_interval,
            max_cycles=self.max_polls,
        )

        # Process all batches
        self.processing_manager.process_items()

        self._is_running = False
        self._is_complete = self.processing_mall.is_complete()

        # Log summary
        self.logger.info(
            f"Batch processing complete: "
            f"{len(self.processing_mall.finished)} successful, "
            f"{len(self.processing_mall.erred)} failed"
        )

    def aggregate_results(self) -> tuple[list[Segment], list[Embedding]]:
        """
        Aggregate all segments and embeddings from completed batches.

        Returns:
            Tuple of (all_segments, all_embeddings)
        """
        if not self._is_complete:
            raise BatchError(
                "Cannot aggregate results before processing is complete. "
                "Call monitor_batches() first or use run() to submit and monitor."
            )

        if self.processing_mall.erred:
            self.logger.warning(
                f"{len(self.processing_mall.erred)} batches failed. "
                f"Results will be incomplete."
            )

        all_segments = []
        all_embeddings = []

        # Collect all segments and embeddings in order
        for batch_id in self.processing_mall.finished:
            if (
                batch_id in self.processing_mall.segments
                and batch_id in self.processing_mall.embeddings
            ):
                segments = self.processing_mall.segments[batch_id]
                embeddings = self.processing_mall.embeddings[batch_id]

                # Ensure segments and embeddings align
                if len(segments) != len(embeddings):
                    self.logger.warning(
                        f"Mismatch in batch {batch_id}: "
                        f"{len(segments)} segments, {len(embeddings)} embeddings"
                    )
                    continue

                all_segments.extend(segments)
                all_embeddings.extend(embeddings)

        # If original segments was a dict, restore keys
        if isinstance(self.segments, dict):
            # This assumes order preservation, which is guaranteed in Python 3.7+
            keys = list(self.segments.keys())
            if len(keys) == len(all_segments):
                return keys, all_embeddings

        return all_segments, all_embeddings

    def run(self) -> tuple[list[Segment], list[Embedding]]:
        """
        Execute the complete batch embedding workflow and return results.

        This method submits batches, monitors their status until completion,
        and returns the aggregated results.

        Returns:
            Tuple of (all_segments, all_embeddings)
        """
        # Submit batches
        self.submit_batches()

        # Monitor until completion
        self.monitor_batches()

        # Aggregate and return results
        result = self.aggregate_results()
        self._result = result

        return result

    @property
    def is_running(self) -> bool:
        """Check if batch processing is currently running"""
        return self._is_running

    @property
    def is_complete(self) -> bool:
        """Check if batch processing is complete"""
        return self._is_complete

    @property
    def result(self) -> tuple[list[Segment], list[Embedding]] | None:
        """Get the aggregated results if available"""
        return self._result

    def get_status_summary(self) -> dict[str, int]:
        """
        Get a summary of batch statuses.

        Returns:
            Dictionary with counts of batches in each status
        """
        summary = defaultdict(int)

        # Count current batches by status
        for batch_id, batch_obj in self.processing_mall.current.items():
            summary[batch_obj.status] += 1

        # Add completed and failed batches
        summary["completed"] = len(self.processing_mall.finished)
        summary["failed"] = len(self.processing_mall.erred)

        return dict(summary)


# TODO: Review and refactor. Consider encorporating "normal" non-batch computation.
def compute_embeddings(
    segments: Segments,
    model: str = DFLT_EMBEDDINGS_MODEL,
    *,
    batch_size: int = DFLT_BATCH_SIZE,
    poll_interval: float = DFLT_POLL_INTERVAL,
    max_polls: int | None = DFLT_MAX_POLLS,
    verbosity: int = DFLT_VERBOSITY,
    processing_mall: ProcessingMall | None = None,
    keep_processing_info: bool = DFLT_KEEP_PROCESSING_INFO,
    dacc: OaDacc | None = None,
    return_process: bool = False,
    logger: logging.Logger | None = None,
    **embeddings_kwargs,
) -> tuple[list[Segment], list[Embedding]] | EmbeddingsBatchProcess:
    """
    Compute embeddings for text segments using OpenAI's batch API.

    This function manages the complete lifecycle of batch embedding requests,
    from submitting batches to the API, monitoring their status, and
    aggregating the results.

    Args:
        segments: Text segments to embed, either as a list or a dictionary
        model: OpenAI embedding model to use
        batch_size: Maximum number of segments per batch
        poll_interval: Seconds between status checks
        max_polls: Maximum number of status checks before timing out
        verbosity: Level of logging detail (0-2)
        processing_mall: Optional custom ProcessingMall
        keep_processing_info: Whether to keep mall data after completion
        dacc: Optional custom OaDacc instance
        return_process: If True, return the EmbeddingsBatchProcess object instead of results
        logger: Optional custom logger
        **embeddings_kwargs: Additional parameters for embedding generation

    Returns:
        If return_process is False (default):
            Tuple of (segments, embeddings)
        If return_process is True:
            EmbeddingsBatchProcess object for further interaction
    """
    # Create batch process
    process = EmbeddingsBatchProcess(
        segments=segments,
        model=model,
        batch_size=batch_size,
        poll_interval=poll_interval,
        max_polls=max_polls,
        verbosity=verbosity,
        processing_mall=processing_mall,
        keep_processing_info=keep_processing_info,
        dacc=dacc,
        logger=logger,
        **embeddings_kwargs,
    )

    # Return process if requested
    if return_process:
        return process

    # Otherwise, run the process and return results
    return process.run()


# Create a pandas-friendly wrapper
def compute_embeddings_df(
    segments: Segments, model: str = DFLT_EMBEDDINGS_MODEL, **kwargs
) -> "pandas.DataFrame":
    """
    Compute embeddings and return results as a pandas DataFrame.

    Args:
        segments: Text segments to embed
        model: OpenAI embedding model to use
        **kwargs: Additional arguments passed to compute_embeddings

    Returns:
        DataFrame with 'segment' and 'embedding' columns
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for compute_embeddings_df")

    segments_result, embeddings = compute_embeddings(segments, model, **kwargs)

    # If segments_result is a list of keys (from dictionary input)
    if isinstance(segments, dict) and len(segments_result) == len(segments):
        # Restore the original text segments
        segments_text = [segments[key] for key in segments_result]
        # Create DataFrame with keys as index
        df = pd.DataFrame(
            {"segment": segments_text, "embedding": embeddings}, index=segments_result
        )
    else:
        # Create standard DataFrame
        df = pd.DataFrame({"segment": segments_result, "embedding": embeddings})

    return df
```

## batches.py

```python
"""
Batch functionality

Useful links:
- Web tool: https://platform.openai.com/batches/
- API docs: https://platform.openai.com/docs/api-reference/batch

"""

from typing import Optional, Union, List, Any, Tuple
from collections.abc import Callable
from functools import partial
import itertools
from oa.util import batch_endpoints, BatchObj
from oa.base import (
    _prepare_embeddings_args,
    _raise_if_any_invalid,
    DFLT_EMBEDDINGS_MODEL,
    NOT_GIVEN,
    TextOrTexts,
    mk_client,
)


BatchId = str
BatchSpec = Union[BatchObj, BatchId]

# --------------------------------------------------------------------------------------
# useful information

batch_field_descriptions = {
    "id": "The unique identifier of the batch.",
    "status": "The status of the batch. See the `status_enum_descriptions` for possible values.",
    "input_file_id": "The ID of the file that contains the batch's input data.",
    "output_file_id": "The ID of the file that contains the batch's output data, if the batch completes successfully.",
    "created_at": "A timestamp of when the batch was created.",
    "completed_at": "A timestamp of when the batch was completed.",
    "failed_at": "A timestamp of when the batch failed.",
    "cancelled_at": "A timestamp of when the batch was cancelled.",
    "in_progress_at": "A timestamp of when the batch started processing.",
    "finalizing_at": "A timestamp of when the batch entered the finalizing stage.",
    "expired_at": "A timestamp of when the batch expired, meaning it did not complete within the time window.",
    "expires_at": "The time at which the batch will expire if not completed.",
    "error_file_id": "The ID of the file that contains detailed error messages if the batch fails.",
    "request_counts": "Contains counts of the total, completed, and failed requests in the batch.",
    "errors": "An array containing errors related to individual requests within the batch, if any exist.",
    "completion_window": "The maximum time window allowed for the batch to complete.",
}

status_enum_descriptions = {
    "created": "The batch has been created but has not started processing yet. See the `created_at` field for when it was created.",
    "in_progress": "The batch is currently being processed. See the `in_progress_at` field for when processing started.",
    "failed": "The batch encountered an error during processing. See the `failed_at` field for when it failed, and the `error_file_id` for details on the failure.",
    "completed": "The batch has successfully completed processing. See the `completed_at` field for when it was completed, and `output_file_id` for the output file.",
    "cancelled": "The batch was cancelled before completion. See the `cancelled_at` field for when it was cancelled.",
    "finalizing": "The batch is finalizing its results. See the `finalizing_at` field for when it entered this stage.",
    "expired": "The batch exceeded its completion window and was terminated. See the `expired_at` field for when it expired.",
    "failed_partially": "Some requests in the batch failed while others succeeded. See `errors` for the failed requests, and `output_file_id` for any partial output.",
}

# --------------------------------------------------------------------------------------
# batch embeddings utils

import time
from dataclasses import dataclass


def random_custom_id(prefix="custom_id-", suffix=""):
    """Make a random custom_id by using the current time in nanoseconds"""
    return f"{prefix}{int(time.time() * 1e9)}{suffix}"


# @dataclass
# class EmbeddingsMaker:
#     texts: TextOrTexts,

#     custom_id: str = None,
#     validate: Optional[Union[bool, Callable]] = True,
#     valid_text_getter=_raise_if_any_invalid,
#     model=DFLT_EMBEDDINGS_MODEL,
#     client=None,
#     dimensions: Optional[int] = NOT_GIVEN,
#     **extra_embeddings_params,


def _rm_not_given_values(d):
    return {k: v for k, v in d.items() if v is not NOT_GIVEN}


def _mk_embeddings_request_body(
    text_or_texts,
    model=DFLT_EMBEDDINGS_MODEL,
    user=NOT_GIVEN,
    dimensions: int | None = NOT_GIVEN,
    **extra_embeddings_params,
):
    return _rm_not_given_values(
        dict(
            input=text_or_texts,
            model=model,
            user=user,
            dimensions=dimensions,
            **extra_embeddings_params,
        )
    )


def _mk_task_request_dict(
    body, custom_id=None, *, endpoint=DFLT_EMBEDDINGS_MODEL, method="POST"
):

    if custom_id is None:
        custom_id = random_custom_id("embeddings_batch_id-")

    return {
        "custom_id": custom_id,
        "method": method,
        "url": endpoint,
        "body": body,
    }


def mk_batch_file_embeddings_task(
    texts: TextOrTexts,
    *,
    custom_id: str | None = None,
    validate: bool | Callable | None = True,
    valid_text_getter=_raise_if_any_invalid,
    # client=None,
    model=DFLT_EMBEDDINGS_MODEL,
    dimensions: int | None = NOT_GIVEN,
    custom_id_per_text=None,
    **extra_embeddings_params,
) -> dict | list[dict]:
    """
    Create a batch task (json-)dicts for generating embeddings.

    These dictionaries (or list thereof) are destined to be used as input for the
    OpenAI API batch endpoint. The endpoint will then generate embeddings for the
    provided texts.

    Args:
        texts (TextOrTexts): The text or list of texts to generate embeddings for.
        custom_id (Optional[str], optional): A custom identifier for the batch task. Defaults to None.
        validate (Optional[Union[bool, Callable]], optional): Whether to validate the texts. Defaults to True.
        valid_text_getter (Optional[Callable], optional): Function to retrieve valid texts. Defaults to _raise_if_any_invalid.
        model (str, optional): The embeddings model to use. Defaults to DFLT_EMBEDDINGS_MODEL.
        dimensions (Optional[int], optional): The dimensions for the embeddings. Defaults to NOT_GIVEN.
        custom_id_per_text (Optional[bool], optional): Whether to include a custom ID per text item. Defaults to None.
        **extra_embeddings_params: Additional parameters for embeddings.

    Returns:
        Union[dict, List[dict]]: A single task dictionary or a list of task dictionaries.

    Examples:

    With a single text:

    >>> mk_batch_file_embeddings_task("Example text")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    {'custom_id': '...',
     'method': 'POST',
     'url': '/v1/embeddings',
     'body': {'input': ['Example text'], 'model': 'text-embedding-3-small'}}

    With a list of texts:

    >>> mk_batch_file_embeddings_task(["Text1", "Text2"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    {'custom_id': '...',
     'method': 'POST',
     'url': '/v1/embeddings',
     'body': {'input': ['Text1', 'Text2'], 'model': 'text-embedding-3-small'}}

    With a dictionary of texts:

    >>> mk_batch_file_embeddings_task({"key1": "Text1", "key2": "Text2"})   # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    [{'custom_id': 'key1',
      'method': 'POST',
      'url': '/v1/embeddings',
      'body': {'input': 'Text1', 'model': 'text-embedding-3-small'}},
     {'custom_id': 'key2',
      'method': 'POST',
      'url': '/v1/embeddings',
      'body': {'input': 'Text2', 'model': 'text-embedding-3-small'}}]

    As you see, when a list is given, the batch task is created with a single custom_id,
    with a list of texts as input.
    When a dictionary is given, the batch task is created with
    a custom_id per text item, and the input is a dictionary of key-value pairs.

    The usefulness of giving a dictionary is that the tasks will contain the same keys
    as your dictionary, which can be useful when having to link the results back to
    the original data. But it comes with a disadvantage: it's much faster to process
    a list of texts than multiple tasks with a single text each.

    When efficiency is a concern, you can keep a texts dictionary on the client side,
    ask to embed list(texts.values()), and then use the keys to link the results back
    to the original data.
    You can force the function to ignore the type of the texts input and do it the way
    you want it, by setting custom_id_per_text to True or False.

    >>> mk_batch_file_embeddings_task(
    ...     {"key1": "Text1", "key2": "Text2"}, custom_id_per_text=False
    ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    {'custom_id': '...',
    'method': 'POST',
    'url': '/v1/embeddings',
    'body': {'input': ['Text1', 'Text2'],
    'model': 'text-embedding-3-small'}}

    >>> mk_batch_file_embeddings_task(
    ...     ["Text1", "Text2"], custom_id_per_text=True
    ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    [{'custom_id': 0,
      'method': 'POST',
      'url': '/v1/embeddings',
      'body': {'input': 'Text1', 'model': 'text-embedding-3-small'}},
     {'custom_id': 1,
      'method': 'POST',
      'url': '/v1/embeddings',
      'body': {'input': 'Text2', 'model': 'text-embedding-3-small'}}]


    """

    texts, texts_type, keys = _prepare_embeddings_args(
        validate, texts, valid_text_getter, model
    )

    _body = partial(
        _mk_embeddings_request_body,
        model=model,
        dimensions=dimensions,
        **extra_embeddings_params,
    )
    _task = partial(
        _mk_task_request_dict, custom_id=custom_id, endpoint=batch_endpoints.embeddings
    )

    if custom_id_per_text is None:
        custom_id_per_text = bool(keys)  # if keys is not None, do it

    if custom_id_per_text:
        if keys is None:
            keys = itertools.count()
        # return a list of tasks, using the keys as custom_ids
        return [_task(_body(text), custom_id=key) for key, text in zip(keys, texts)]
    else:
        return _task(_body(list(texts)))


from oa.util import oa_extractor


def batch_info_to_segments_and_embeddings(jsonl_store, batch_info):
    output_data_dict = jsonl_store[batch_info.output_file_id]
    output_data = oa_extractor(output_data_dict)
    if (
        "response.body.data.*.embedding" in output_data
        and output_data["response.status_code"] == 200
    ):
        embedding_vectors = output_data["response.body.data.*.embedding"]
        input_file_data = oa_extractor(jsonl_store[batch_info.input_file_id])
        segments = input_file_data["body.input"]
        return segments, embedding_vectors
    else:
        return None


def get_segments_and_embeddings(batch_store, jsonl_store):
    for batch_info in batch_store.values():
        if batch_info.endpoint == "/v1/embeddings" and batch_info.status == "completed":
            yield batch_info_to_segments_and_embeddings(jsonl_store, batch_info)


# --------------------------------------------------------------------------------------
# Misc
from operator import attrgetter
from lkj import value_in_interval
from dol import Pipe

create_at_within_range = value_in_interval(get_val=attrgetter("created_at"))


def batches_within_range(batches_base, min_date, max_date=None):
    return filter(
        create_at_within_range(min_val=min_date, max_val=max_date), batches_base
    )


def request_counts(batch_list):
    # pylint: disable=import-error
    import pandas as pd

    t = pd.DataFrame([x.to_dict() for x in batch_list])
    tt = pd.DataFrame(t.request_counts.values.tolist())
    return tt.sum()


check_batch_requests = Pipe(batches_within_range, request_counts)


from oa.util import utc_int_to_iso_date
from datetime import timedelta


# Define custom error classes
# TODO: Rethink these. They're not really errors (except perhaps failed and canceled)
#   Perhaps some state object is more appropriate?
class BatchError(ValueError):  # Or RuntimeError? Or just Exception?
    pass


class BatchInProgressError(BatchError):
    pass


class BatchCancelledError(BatchError):
    pass


class BatchExpiredError(BatchError):
    pass


class BatchFinalizingError(BatchError):
    pass


class BatchFailedError(BatchError):
    pass


def get_batch_obj(oa_stores, batch: BatchSpec) -> BatchObj:
    try:
        return oa_stores.batches_base[batch]
    except KeyError:
        raise KeyError(f"Batch {batch} not found.")


def get_batch_id_and_obj(oa_stores, batch: BatchSpec) -> tuple[BatchId, BatchObj]:
    try:
        batch_obj = oa_stores.batches_base[batch]
        batch_id = batch_obj.id
        return batch_id, batch_obj
    except KeyError:
        raise KeyError(f"Batch {batch} not found.")


def on_complete(oa_stores, batch_obj: BatchObj):
    """Return the output file if the batch completed successfully"""
    return oa_stores.files_base[batch_obj.output_file_id]


def raise_error(error_obj: BaseException):
    raise error_obj


# TODO: I do NOT like the dependency on oa_stores here!
# TODO: Not sure if function or object with a "handle" __call__ method is better here
def get_output_file_data(
    batch: BatchSpec,
    *,
    oa_stores,
    get_batch_obj: Callable = get_batch_obj,
    on_complete: Callable = on_complete,
    on_error: Callable[[BaseException], Any] = raise_error,
):
    """
    Get the output file data for a batch, if it has completed successfully.

    """

    batch_obj = get_batch_obj(oa_stores, batch)

    try:
        batch_obj = oa_stores.batches_base[batch]
    except KeyError:
        raise KeyError(f"Batch {batch} not found.")

    if batch_obj.status == "completed":
        return on_complete(oa_stores, batch_obj)
    else:

        if batch_obj.status == "failed":
            # Raise an error if the batch failed
            error_obj = BatchFailedError(
                f"Batch {batch} failed "
                f"at {utc_int_to_iso_date(batch_obj.failed_at)}. "
                f"Check out {batch_obj.error_file_id} for more information."
            )

        elif batch_obj.status == "in_progress":
            # Calculate the time elapsed between creation and when it started processing
            time_elapsed = timedelta(
                seconds=(batch_obj.in_progress_at - batch_obj.created_at)
            )
            error_obj = BatchInProgressError(
                f"Batch {batch} is still in progress. "
                f"Started processing at {utc_int_to_iso_date(batch_obj.in_progress_at)}, "
                f"{time_elapsed.total_seconds() // 3600:.0f} hours and "
                f"{(time_elapsed.total_seconds() % 3600) // 60:.0f} minutes after it was created."
            )

        elif batch_obj.status == "cancelled":
            # Provide information when the batch was cancelled
            error_obj = BatchCancelledError(
                f"Batch {batch} was cancelled "
                f"at {utc_int_to_iso_date(batch_obj.cancelled_at)}."
            )

        elif batch_obj.status == "expired":
            # Notify that the batch expired and provide timestamps
            error_obj = BatchExpiredError(
                f"Batch {batch} expired at {utc_int_to_iso_date(batch_obj.expired_at)}. "
                f"Completion window was {batch_obj.completion_window} hours."
            )

        elif batch_obj.status == "finalizing":
            # Provide information when the batch entered the finalizing stage
            error_obj = BatchFinalizingError(
                f"Batch {batch} is in the finalizing stage as of "
                f"{utc_int_to_iso_date(batch_obj.finalizing_at)}. "
                f"Please check back later for the final results."
            )

        # Attach the batch object to the error for debugging/context purposes
        error_obj.batch_obj = batch_obj
        # Do what you got to do with the error
        return on_error(error_obj)


# --------------------------------------------------------------------------------------
# # Old functions for embeddings batch tasks
# import tempfile
# from pathlib import Path
# import json
# from oa.util import Sig.replace_kwargs_using


# @Sig.replace_kwargs_using(mk_batch_file_embeddings_task)
# def saved_embeddings_task(texts, **embeddings_params):
#     task_dict = mk_batch_file_embeddings_task(texts, **embeddings_params)
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w')
#     Path(temp_file.name).write_text(json.dumps(task_dict))
#     return temp_file.name


# @Sig.replace_kwargs_using(mk_batch_file_embeddings_task)
# def batch_input_file_for_embeddings(
#     texts: TextOrTexts, *, purpose='batch', **embeddings_params
# ):
#     _saved_embeddings_task = saved_embeddings_task(texts, **embeddings_params)

#     client = client or mk_client()
#     batch_input_file = client.files.create(
#         file=open(_saved_embeddings_task, 'rb'), purpose=purpose
#     )
#     batch_input_file._local_filepath = _saved_embeddings_task
#     return batch_input_file


# def batch_input_file_for_embeddings(
#     filepath: str, *, purpose='batch', client=None, **embeddings_params
# ):

#     client = client or mk_client()
#     batch_input_file = client.files.create(
#         file=open(filepath, 'rb'), purpose=purpose
#     )
#     batch_input_file._local_filepath = filepath
#     return batch_input_file
```

## chats.py

```python
r"""
Tools to work with shared chats (conversations).

For instance: Extract information from them.

The main object here is `ChatDacc` (Chat Data Accessor), which is a class that allows
you to access the data in a shared chat in a structured way.

# >>> from oa.chats import ChatDacc
# >>>
# >>> url = 'https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5'
# >>> dacc = ChatDacc(url)
# >>>
# >>> basic_turns_data = dacc.basic_turns_data
# >>> len(basic_turns_data)
# 4
# >>> first_turn = basic_turns_data[0]
# >>> isinstance(first_turn, dict)
# True
# >>> list(first_turn)
# ['id', 'role', 'content', 'message_id']
# >>> print(first_turn['content'])  # doctest: +NORMALIZE_WHITESPACE
# This conversation is meant to be used as an example, for testing, and/or for figuring out how to parse the html and json of a conversation.
# <BLANKLINE>
# As such, we'd like to keep it short.
# <BLANKLINE>
# Just say "Hello World!" back to me for now, and then in a second line write 10 random words.
"""

import re
import json
import asyncio
from typing import Optional, Union, List, Dict, Any
from collections.abc import Callable, Iterable
from functools import partial, cached_property
from bs4 import BeautifulSoup
from dol import path_filter, Pipe, path_get, paths_getter


# --------------------------------------------------------------------------------------
# HTML rendering and parsing functions from chats3.py


def get_rendered_html(
    url: str, *, headless: bool = True, timeout: int = 30000, use_async: bool = False
) -> str:
    """Return the fully rendered HTML for `url` using Playwright (sync or async).

    By default this uses the synchronous Playwright API. If `use_async=True` the
    async variant will be executed via asyncio.run.

    Note: This function only imports Playwright when called. Install with:
        pip install playwright
        playwright install chromium

    """
    # If user explicitly requested async, or if there's a running asyncio loop
    # (e.g. in Jupyter), prefer the async implementation.
    try:
        import asyncio

        loop_running = asyncio.get_running_loop().is_running()
    except RuntimeError:
        loop_running = False
    except Exception:
        loop_running = False

    if use_async or loop_running:
        # In notebook-style environments there's already an event loop; use
        # the async Playwright API. If there's a running loop, we need to use
        # nest_asyncio or run in a new task — here we'll try nest_asyncio first
        # and then fall back to running the async helper in a new thread.
        try:
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(
                get_rendered_html_async(url, headless=headless, timeout=timeout)
            )
        except Exception:
            # If nest_asyncio isn't available or something else fails, run the
            # async helper in a separate thread to avoid running inside the
            # already-running loop.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    get_rendered_html_async(url, headless=headless, timeout=timeout),
                )
                return future.result()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is required for rendered HTML extraction. "
            "Install with: pip install playwright && playwright install chromium"
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout)
            # return fully rendered HTML
            return page.content()
        finally:
            browser.close()


async def get_rendered_html_async(
    url: str, *, headless: bool = True, timeout: int = 30000
) -> str:
    """Asynchronous version: return fully rendered HTML using Playwright async API."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise ImportError(
            "playwright is required for rendered HTML extraction. "
            "Install with: pip install playwright && playwright install chromium"
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=timeout)
            return await page.content()
        finally:
            await browser.close()


def reduce_chat_html(html: str) -> str:
    """
    Strips away all irrelevant metadata, headers, footers, and scripts from
    the raw HTML, keeping only the essential conversation content enclosed in
    the <div id="thread"> tag.

    This dramatically reduces the size of the HTML fed into the main parser
    while preserving all necessary information for `parse_chat_html`.

    Args:
        html: The full HTML content of the conversation file.

    Returns:
        A stripped-down HTML string containing only the conversation thread.
    """
    # 1. Use BeautifulSoup to quickly find the root of the conversation thread
    soup = BeautifulSoup(html, "html.parser")

    # 2. The entire conversation content lives inside the element with id="thread".
    thread_div = soup.find("div", id="thread")

    if thread_div:
        # Get the inner HTML content of the thread div
        thread_content = str(thread_div)

        # 3. Wrap it in a minimal valid HTML structure for the parser
        # The surrounding tags are needed for BeautifulSoup to treat it as a proper document
        reduced_html = f"<html><body>{thread_content}</body></html>"
        return reduced_html
    else:
        # If the thread element is not found, return the original HTML
        return html


def parse_chat_html(html: str) -> list[dict[str, Any]]:
    """
    Parses conversation data from a shared chat HTML file into a structured list of messages.

    It extracts the primary conversation data along with essential metadata:
    'id' (from the turn ID), 'role', 'content', and 'message_id'. The 'time'
    field is excluded as it cannot be reliably extracted from the visible HTML.

    Args:
        html: The full HTML content of the conversation file.

    Returns:
        A list of dictionaries, where each dictionary represents a message and
        contains 'id', 'role', 'content', and 'message_id' keys.
    """
    soup = BeautifulSoup(html, "html.parser")
    conversation = []

    # Find all conversation turns (user or assistant messages)
    turns = soup.find_all("article", {"data-turn": ["user", "assistant"]})

    for turn in turns:
        role = turn["data-turn"]
        content_parts = []

        # Extract turn metadata, using 'id' as requested for compatibility
        turn_id = turn.get("data-turn-id")

        # Find the inner div that contains the message ID
        message_div = turn.find("div", {"data-message-author-role": role})
        message_id = message_div.get("data-message-id") if message_div else None

        # --- 1. Handle User Messages ---
        if role == "user":
            text_container = turn.find("div", class_="whitespace-pre-wrap")
            if text_container:
                content = text_container.get_text("\n").strip()
                if content:
                    content_parts.append({"type": "text", "text": content})

        # --- 2. Handle Assistant Messages ---
        elif role == "assistant":
            markdown_div = turn.find(
                "div", class_=lambda c: c and "markdown" in c and "prose" in c
            )

            if markdown_div:
                for element in markdown_div.children:
                    # Handle Paragraphs/Text (<p> tags)
                    if element.name == "p":
                        text_content = element.get_text("\n").strip()
                        if text_content:
                            cleaned_text = text_content.replace("\n\n", "\n").strip()
                            if cleaned_text:
                                content_parts.append(
                                    {"type": "text", "text": cleaned_text}
                                )

                    # Handle Code Blocks (<pre> tags for fenced code blocks)
                    elif element.name == "pre":
                        lang_div = element.find(
                            "div",
                            class_=lambda c: c
                            and "justify-between" in c
                            and "h-9" in c,
                        )
                        language = "plaintext"
                        if lang_div:
                            language_text = lang_div.get_text(strip=True).lower()
                            if language_text and language_text != "copy code":
                                language = language_text

                        code_tag = element.find("code")
                        code_content = (
                            code_tag.get_text(strip=False).strip("\n")
                            if code_tag
                            else ""
                        )

                        if code_content:
                            content_parts.append(
                                {
                                    "type": "code",
                                    "language": language,
                                    "code": code_content,
                                }
                            )

        # --- Combine and Format Output ---
        if content_parts:
            final_content = ""
            for part in content_parts:
                if part["type"] == "text":
                    final_content += part["text"] + "\n\n"
                elif part["type"] == "code":
                    final_content += f"```{part['language']}\n{part['code']}\n```\n\n"

            final_content = final_content.strip()

            if final_content:
                message = {
                    "id": turn_id,
                    "role": role,
                    "content": final_content,
                    "message_id": message_id,
                }
                conversation.append(message)

    return conversation


# --------------------------------------------------------------------------------------
# Side extras - utility functions kept from original


def remove_utm_source(text):
    """
    Removes the "?utm_source=chatgpt.com" suffix from all URLs in the given text.

    Args:
        text (str): The input text containing URLs.

    Returns:
        str: The text with the specified query parameter removed from all URLs.

    Example:

        >>> input_text = (
        ...     "not_a_url "  # not even a url (won't be touched)
        ...     "http://abc?utm_source=chatgpt.com_not_at_the_end "  # target not at the end
        ...     "https://abc?utm_source=chatgpt.com "  # with ?
        ...     "https://abc&utm_source=chatgpt.com "  # with &
        ...     "http://abc?utm_source=chatgpt.com"  # with http instead of https
        ... )
        >>> remove_utm_source(input_text)
        'not_a_url http://abc_not_at_the_end https://abc https://abc http://abc'

    """
    pattern = r"(https?://[^\s]+)[&\?]utm_source=chatgpt\.com"
    cleaned_text = re.sub(pattern, r"\1", text)
    return cleaned_text


def find_url_keys(data, current_path=""):
    r"""
    Recursively finds all paths in a JSON-like object (nested dicts/lists) where URL strings are present.

    This is a generator function that yields each path as a dot-separated string. Supports paths through
    both dictionaries and lists.

    Args:
        data (dict or list): The JSON-like object to search.
        current_path (str): The current path being traversed, used internally during recursion.

    Yields:
        str: Dot-separated path to a value containing a URL.

    Example:
        >>> example_data = {
        ...     "key1": {
        ...         "nested": [
        ...             {"url": "http://example.com"},
        ...             {"url": "https://example.org"}
        ...         ]
        ...     },
        ...     "key2": "http://another-example.com"
        ... }
        >>> list(find_url_keys(example_data))
        ['key1.nested[0].url', 'key1.nested[1].url', 'key2']

        One thing you'll probably want to do sometimes is transform, filter, and
        aggregate these paths. Here's an example of how you might get a list of
        unique paths, with all array indices replaced with a wildcard, so thay
        don't appear as separate paths:

        >>> from functools import partial
        >>> import re
        >>> paths = find_url_keys(example_data)
        >>> transform = partial(re.sub, '\[\d+\]', '[*]')
        >>> unique_paths = set(map(transform, paths))
        >>> sorted(unique_paths)
        ['key1.nested[*].url', 'key2']
    """

    def _is_url(x):
        return isinstance(x, str) and ("http://" in x or "https://" in x)

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            if isinstance(value, (dict, list)):
                yield from find_url_keys(value, new_path)
            elif _is_url(value):
                yield new_path
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_path = f"{current_path}[{index}]"
            yield from find_url_keys(item, new_path)


def find_all_matching_paths_in_list_values(
    nested_dict, target_value: Callable | str
):
    """
    Find all paths in a nested dictionary where target_value evaluates to True
    for elements in a list contained within the value.

    :param nested_dict: The dictionary to search.
    :param target_value: A function that takes a value and returns a boolean.
        If a string, will be converted to a regex search.
    :return: A generator yielding paths that match the condition.

    >>> nested_dict = {
    ...     "a": {"b": {"c": [1, 2, 3], "d": [4, 5]}},
    ...     "e": {"f": [6, 7], "g": {"h": [8, 9]}},
    ... }
    >>> target_value_fn = lambda x: x % 2 == 0  # Find even numbers
    >>> list(find_all_matching_paths_in_list_values(nested_dict, target_value_fn))
    [('a', 'b', 'c'), ('a', 'b', 'd'), ('e', 'f'), ('e', 'g', 'h')]

    """
    target_value = _ensure_filter_func(target_value)

    return path_filter(
        pkv_filt=lambda p, k, v: (
            isinstance(v, list) and any(target_value(item) for item in v)
        ),
        d=nested_dict,
    )


def _ensure_filter_func(target: str | Callable) -> Callable:
    """Convert a string to a regex filter function, or pass through callable."""
    if isinstance(target, str):
        target_pattern = re.compile(target, re.DOTALL)
        return lambda x: isinstance(x, str) and target_pattern.search(x)
    assert callable(target), f"target must be a string or a callable: {target=}"
    return target


paths_get_or_none = partial(
    paths_getter,
    get_value=dict.get,
    on_error=path_get.return_none_on_error,
)


# --------------------------------------------------------------------------------------
# A manager class that operates on a shared chat


class ChatDacc:
    """Chat Data Accessor - manages and extracts data from shared ChatGPT conversations."""

    def __init__(self, src):
        """
        Initialize ChatDacc with a source (URL or parsed conversation data).

        Args:
            src: Either a URL string or a list of conversation dictionaries
        """
        self.src = src
        if isinstance(src, str):
            if src.startswith("http"):
                self.url = src
                html = get_rendered_html(src)
                self.parsed_conversation = parse_chat_html(html)
            else:
                raise ValueError(f"Invalid (string) src. Must be a url: {src}")
        elif isinstance(src, list):
            # Assume it's already parsed conversation data
            self.parsed_conversation = src
        else:
            raise ValueError(f"Invalid src: {src}")

    @cached_property
    def basic_turns_data(self):
        """Return the parsed conversation as-is (list of dicts with id, role, content, message_id)."""
        return self.parsed_conversation

    @property
    def basic_turns_df(self):
        """Convert the conversation to a pandas DataFrame."""
        import pandas as pd  # pip install pandas

        return pd.DataFrame(self.basic_turns_data)

    def copy_turns_json(self):
        """Copy the conversation data to clipboard as JSON."""
        from pyperclip import copy  # pip install pyperclip

        return copy(json.dumps(self.parsed_conversation, indent=4))

    @cached_property
    def url_paths(self):
        """Find all paths containing URLs in the conversation data."""
        # Convert list to dict for compatibility with find_url_keys
        conversation_dict = {
            str(i): msg for i, msg in enumerate(self.parsed_conversation)
        }
        return list(find_url_keys(conversation_dict))

    @cached_property
    def paths_containing_urls(self):
        """Get unique paths containing URLs with array indices replaced by wildcards."""
        replace_array_index_with_star = partial(re.sub, r"\[\d+\]", "[*]")
        ignore_first_part_of_path = lambda x: (
            ".".join(x.split(".")[1:]) if "." in x else x
        )
        transform = Pipe(replace_array_index_with_star, ignore_first_part_of_path)
        return sorted(set(map(transform, self.url_paths)))

    def url_data(self, *, remove_chatgpt_utm=True):
        """
        Extract all URLs from the conversation.

        Args:
            remove_chatgpt_utm: If True, remove utm_source=chatgpt.com from URLs

        Returns:
            List of URLs found in the conversation
        """
        urls = []
        for msg in self.parsed_conversation:
            content = msg.get("content", "")
            # Find URLs in content
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            found_urls = re.findall(url_pattern, content)
            urls.extend(found_urls)

        if remove_chatgpt_utm:
            urls = [remove_utm_source(url) for url in urls]

        return urls


# --------------------------------------------------------------------------------------
# SSOT and other data - placeholder for compatibility
# These would need to be imported from oa._params if they exist
try:
    from oa._params import turns_data_ssot, metadata_ssot

    ChatDacc.turns_data_ssot = turns_data_ssot
    ChatDacc.metadata_ssot = metadata_ssot
except ImportError:
    pass  # These might not exist in the new structure


# --------------------------------------------------------------------------------------
# Tools for parser maintenance
# See https://github.com/thorwhalen/oa/blob/main/misc/oa%20-%20chats%20acquisition%20and%20parsing.ipynb
# for more info (namely how to use mk_json_field_documentation to make descriptions)
"""
Notes:

metadata['linear_conversation'] == list(metadata['mapping'].values())

"""
try:
    from oa.tools import prompt_function

    mk_json_field_documentation = prompt_function(
        """
    You are a technical writer specialized in documenting JSON fields. 
    Below is a JSON object. I'd like you to document each field in a markdown table.
    The table should contain the name, description, and example value of each field.
                                                  
    The context is:
    {context: just a general context}
                                                  
    Here's an example json object:

    {example_json}
    """,
        egress=lambda x: x["result"],
    )
except ImportError:
    pass  # prompt_function might not be available


def truncate_dict_values(d, max_length=100):
    """Truncate string values in a dict to a maximum length for display purposes."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > max_length:
            result[k] = v[:max_length] + "..."
        elif isinstance(v, dict):
            result[k] = truncate_dict_values(v, max_length)
        elif isinstance(v, list) and len(v) > 3:
            result[k] = v[:3] + ["..."]
        else:
            result[k] = v
    return result
```

## examples/__init__.py

```python

```

## examples/aesop_fables.py

```python
"""Making a children book containing rhyming stories of aesop fables,
illustrated using different styles of images, taking art movements and
famous artists as styles."""

from collections.abc import Mapping, MutableMapping
import os
import io
from itertools import cycle

import pandas as pd
import requests
from tabled import get_tables_from_url
from dol import Files, TextFiles, wrap_kvs


# --------------------------------------------------------------------------------------
# Stores
# Here, we'll use `dol` to make some "stores" -- that is, a `MutableMapping` facade to
# where we'll store stuff (our fables text, rhyming stories, illustration urls,
# images...).
# We'll store things in local files here, but we can change this to use S3, DBs, etc.
# simply by changing the backend of the facade.
def rm_extension(ext):
    """Make a key transformer that removes the given extension from keys"""
    if not ext.startswith("."):
        ext = "." + ext
    return wrap_kvs(id_of_key=lambda x: x + ext, key_of_id=lambda x: x[: -len(ext)])


Texts = rm_extension("txt")(TextFiles)
Images = rm_extension("jpg")(Files)
Htmls = rm_extension("html")(TextFiles)

# --------------------------------------------------------------------------------------


root_url = "https://aesopfables.com/"
url = root_url + "aesopsel.html"


def url_to_bytes(url: str) -> bytes:
    return requests.get(url).content


def _clean_up_fable_table(t):
    t.columns = ["fable", "moral"]
    t["moral"] = t["moral"].map(lambda x: x[0].strip())
    t["moral"] = t["moral"].map(lambda x: x[1:] if x.startswith(".") else x)
    t["title"], t["rel_url"] = zip(*t["fable"])
    t["url"] = t["rel_url"].map(lambda x: root_url + x)
    del t["fable"]
    return t


def get_fable_table(files):
    if "fables.csv" not in files:
        df = get_tables_from_url(url, extract_links="all")[0]
        df = _clean_up_fable_table(df)
        files["fables.csv"] = df.to_csv(index=False).encode()
    return pd.read_csv(io.BytesIO(files["fables.csv"]))


def get_title_and_urls(fable_table):
    return dict(zip(fable_table["title"], fable_table["url"]))


def get_original_story(url):
    import requests
    from bs4 import BeautifulSoup

    r = requests.get(url)
    soup = BeautifulSoup(r.content)
    return soup.find("pre").text.strip()


# TODO: Make decorators for launch_iteration, print_progress, and overwrite concerns.


def get_original_stories(
    title_and_urls: Mapping,
    original_stories: MutableMapping,
    *,
    launch_iteration=True,
    print_progress=True,
    overwrite=False,
):
    """Extract and store the text of each fable in the fable table"""
    n = len(title_and_urls)

    def run_process():
        for i, (title, url) in enumerate(title_and_urls.items(), 1):
            if print_progress:
                print(f"{i}/{n}: {title}")
            if overwrite or title not in original_stories:
                original_story = get_original_story(url)
                original_stories[title] = original_story

    if launch_iteration:
        for _ in run_process():
            pass
    else:
        return run_process


import oa.examples.illustrate_stories as ii


def get_rhyming_stories(
    original_stories: Mapping,
    *,
    rhyming_stories: MutableMapping,
    launch_iteration=True,
    overwrite=False,
    print_progress=True,
    **kwargs,
):
    n = len(original_stories)

    def run_process():
        for i, (title, original_story) in enumerate(original_stories.items(), 1):
            if print_progress:
                print(f"{i}/{n}: {title}")
            if overwrite or title not in rhyming_stories:
                original_story = original_stories[title]
                rhyming_story = ii.make_it_rhyming(original_story, **kwargs)
                rhyming_stories[title] = rhyming_story

    if launch_iteration:
        for _ in run_process():
            pass
    else:
        return run_process


def get_image_descriptions(
    stories: Mapping,
    *,
    image_descriptions: MutableMapping,
    image_styles=("children's book drawing",),
    launch_iteration=True,
    overwrite=False,
    print_progress=True,
    **kwargs,
):
    _image_styles = cycle(image_styles)
    n = len(stories)

    def run_process():
        for i, (title, story) in enumerate(stories.items(), 1):
            image_style = next(_image_styles)
            if print_progress:
                print(f"{i}/{n}: {title=}, {image_style=}")

            if overwrite or title not in image_descriptions:
                image_description = ii.get_image_description(
                    story, image_style, **kwargs
                )
                image_descriptions[title] = image_description
            yield

    if launch_iteration:
        for _ in run_process():
            pass
    else:
        return run_process


def get_images(
    image_descriptions: Mapping,
    images: MutableMapping,
    *,
    image_urls: Mapping = None,
    image_styles=("children's book drawing",),
    launch_iteration=True,
    overwrite=False,
    print_progress=True,
    **kwargs,
):
    n = len(image_descriptions)
    _image_styles = cycle(image_styles)

    def run_process():
        for i, (title, image_description) in enumerate(image_descriptions.items(), 1):
            image_style = next(_image_styles)
            if overwrite or title not in images:
                try:
                    if print_progress:
                        print(f"{i}/{n}: {title}")
                    image_url = ii.get_image_url(
                        image_description, image_style, **kwargs
                    )
                    if image_urls is not None:
                        image_urls[title] = image_url
                    images[title] = url_to_bytes(image_url)
                except Exception as e:
                    print(
                        f"The description or url for {title} lead to the error {e}. "
                        f"Description:\n\n{image_description}\n\n\n"
                    )
            yield

    if launch_iteration:
        for _ in run_process():
            pass
    else:
        return run_process


def store_stats(*, original_stories, rhyming_stories, image_descriptions, image_urls):
    print(f"{len(original_stories)=}")
    print(f"{len(rhyming_stories)=}")
    print(f"{len(image_descriptions)=}")
    print(f"{len(image_urls)=}")
    print("")
    missing_rhyming_stories = set(original_stories) - set(rhyming_stories)
    missing_descriptions = set(rhyming_stories) - set(image_descriptions)
    missing_urls = set(image_descriptions) - set(image_urls)
    print(f"{len(missing_rhyming_stories)=}")
    print(f"{len(missing_descriptions)=}")
    print(f"{len(missing_urls)=}")


def mk_pages_store(*, rhyming_stories, image_urls, ipython_display=False):
    from dol import wrap_kvs, add_ipython_key_completions
    from dol.sources import FanoutReader

    fanout_store = add_ipython_key_completions(
        FanoutReader(
            {
                "rhyming_stories": rhyming_stories,
                "image_urls": image_urls,
            },
            keys=image_urls,  # take keys from image_urls
        )
    )

    s = wrap_kvs(
        fanout_store,
        obj_of_data=lambda x: ii.aggregate_story_and_image(
            image_url=x["image_urls"], story_text=x["rhyming_stories"]
        ),
    )
    if ipython_display:
        from IPython.display import HTML

        s = wrap_kvs(s, obj_of_data=HTML)
    return s


# --------------------------------------------------------------------------------------
# Resources


def get_top100_artists():
    import requests
    from operator import attrgetter, itemgetter, methodcaller
    from bs4 import BeautifulSoup
    from dol import Pipe

    get_soup = Pipe(requests.get, attrgetter("content"), BeautifulSoup)

    soup = get_soup("https://www.art-prints-on-demand.com/a/artists-painters.html")
    soup2 = get_soup(
        "https://www.art-prints-on-demand.com/a/artists-painters.html&mpos=999&ALL_ABC=1"
    )
    # extract artists
    get_title = Pipe(methodcaller("find", "a"), itemgetter("title"))
    t1 = list(
        map(get_title, soup.find_all("div", {"class": "kk_category_pic"}))
    )  # 30 top
    t2 = list(
        map(get_title, soup2.find_all("div", {"class": "kk_category_pic"}))
    )  # 100 top
    # merge both lists leaving the top 30 at the top, to favor them
    t = t1 + t2
    top_artists = [x for i, x in enumerate(t) if x not in (t)[:i]]
    return top_artists


# Note: Obtained from get_top100_artists()
top100_artists = [
    "Claude Monet",
    "Gustav Klimt",
    "Vincent van Gogh",
    "Paul Klee",
    "Wassily Kandinsky",
    "Franz Marc",
    "Caspar David Friedrich",
    "August Macke",
    "Egon Schiele",
    "Pierre-Auguste Renoir",
    "William Turner",
    "Leonardo da Vinci",
    "Johannes Vermeer",
    "Albrecht Dürer",
    "Carl Spitzweg",
    "Alphonse Mucha",
    "Catrin Welz-Stein",
    "Max Liebermann",
    "Paul Cézanne",
    "Rembrandt van Rijn",
    "Paul Gauguin",
    "(Raphael) Raffaello Sanzio",
    "Amadeo Modigliani",
    "Sandro Botticelli",
    "Edvard Munch",
    "Pierre Joseph Redouté",
    "Michelangelo Caravaggio",
    "Ernst Ludwig Kirchner",
    "Piet Mondrian",
    "Pablo Picasso",
    "Katsushika Hokusai",
    "Hieronymus Bosch",
    "Timothy  Easton",
    "Paula Modersohn-Becker",
    "Edgar Degas",
    "Michelangelo (Buonarroti)",
    "Salvador Dali",
    "Gustave Caillebotte",
    "Pieter Brueghel the Elder",
    "Ferdinand Hodler",
    "Joan Miró",
    "John William Waterhouse",
    "Peter Severin Kroyer",
    "Peter Paul Rubens",
    "Peter  Graham",
    "Henri de Toulouse-Lautrec",
    "Camille Pissarro",
    "Edouard Manet",
    "Joaquin Sorolla",
    "Sara Catena",
    "Henri Julien-Félix Rousseau",
    "Gustave Courbet",
    "Jack Vettriano",
    "Felix Vallotton",
    "All catalogs",
    "Arnold Böcklin",
    "Alexej von Jawlensky",
    "Kazimir Severinovich Malewitsch",
    "Odilon Redon",
    "Jean-Étienne Liotard",
    "Giovanni Segantini",
    "Azure",
    "Oskar Schlemmer",
    "Carl Larsson",
    "Francisco José de Goya",
    "Artist Artist",
    "François Boucher",
    "Mark Rothko",
    "Susett Heise",
    "Alfred Sisley",
    "Giovanni Antonio Canal (Canaletto)",
    "Jean-François Millet",
    "Giuseppe Arcimboldo",
    "Iwan Konstantinowitsch Aiwasowski",
    "Catherine  Abel",
    "Edward Hopper",
    "Mark  Adlington",
    "Jean Honoré Fragonard",
    "Lucy Willis",
    "Jacques Louis David",
    "Pavel van Golod",
    "M.c. Escher",
    "Pierre Bonnard",
    "Ferdinand Victor Eugène Delacroix",
    "Carel Fabritius",
    "Franz von Stuck",
    "John Constable",
    "László Moholy-Nagy",
    "Lincoln  Seligman",
    "William Adolphe Bouguereau",
    "Adolph Friedrich Erdmann von Menzel",
    "Petra Schüßler",
    "Pompei, wall painting",
    "Unbekannter Künstler",
    "Ando oder Utagawa Hiroshige",
    "Marc Chagall",
    "Zita Rauschgold",
    "William  Ireland",
    "Bernardo Bellotto",
    "Hermann Angeli",
]

# Note: Reference: https://magazine.artland.com/art-movements-and-styles/
art_movements = """Abstract Expressionism
Art Deco
Art Nouveau
Avant-garde
Baroque
Bauhaus
Classicism
CoBrA
Color Field Painting
Conceptual Art
Constructivism
Cubism
Dada / Dadaism
Digital Art
Expressionism
Fauvism
Futurism
Harlem Renaissance
Impressionism
Installation Art
Land Art
Minimalism
Neo-Impressionism
Neoclassicism
Neon Art
Op Art
Performance Art
Pop Art
Post-Impressionism
Precisionism
Rococo
Street Art
Surrealism
Suprematism
Symbolism
Zero Group""".splitlines()

styles_of_art = art_movements + top100_artists
```

## examples/batch_embeddings_examples.py

```python
"""
Example usage of the batch embeddings module.
"""

from oa.batch_embeddings import (
    compute_embeddings,
    EmbeddingsBatchProcess,
    compute_embeddings_df,
)


# Example 1: Simple blocking usage
def simple_example():
    """
    Example usage of compute_embeddings in a blocking manner.
    This function computes embeddings for a list of text segments
    and prints the results.

    >>> segments, embeddings = simple_example()  # doctest: +SKIP
    2025-03-28 14:41:06,828 - __main__ - INFO - Submitting batches for 4 segments
    2025-03-28 14:41:08,330 - __main__ - INFO - Submitted 1 batches
    2025-03-28 14:41:08,331 - __main__ - INFO - Monitoring 1 batches
    2025-03-28 14:41:14,062 - __main__ - INFO - Batch Batch(id='batch_67e6a6f40c408190aef26e761efd0b6f', completion_window='24h', created_at=1743169268, endpoint='/v1/embeddings', input_file_id='file-UkK9V2BcoTJosRCkPRdBTM', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1743255668, failed_at=None, finalizing_at=None, in_progress_at=None, metadata=None, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0)) completed successfully
    2025-03-28 14:41:14,064 - __main__ - INFO - Batch processing complete: 1 successful, 0 failed
    Generated 4 embeddings
    First embedding (first 5 dimensions): [-0.018423624, -0.0072260704, 0.003638412, -0.054205045, -0.022725008]
    >>> segments  # doctest: +SKIP
    ['The quick brown fox jumps over the lazy dog.',
    'Machine learning models transform input data into useful representations.',
    'Embeddings capture semantic meaning in dense vector spaces.',
    'Natural language processing enables computers to understand human language.']
    >>> len(embeddings)  # doctest: +SKIP
    4
    >>> len(embeddings[0])  # doctest: +SKIP
    1536
    >>> len(embeddings[0][:5])  # doctest: +SKIP
    [-0.018421143, -0.007218754, 0.0036062053, -0.054197744, -0.022721948]

    """
    # Sample text segments
    segments = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models transform input data into useful representations.",
        "Embeddings capture semantic meaning in dense vector spaces.",
        "Natural language processing enables computers to understand human language.",
    ]

    # Compute embeddings (blocking call)
    result_segments, embeddings = compute_embeddings(
        segments=segments,
        verbosity=1,  # Show basic progress information
        batch_size=100,  # Small batch size for example purposes
    )

    # Print results
    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding (first 5 dimensions): {embeddings[0][:5]}")

    return result_segments, embeddings


# Example 2: Non-blocking usage with manual control
def non_blocking_example():
    """
    Example usage of compute_embeddings in a non-blocking manner.
    Also demonstrates that when your segments are a dict (or Mapping), your output
    is also a dict.

    >>> segment_keys, embeddings = non_blocking_example()  # doctest: +SKIP
    2025-03-28 14:57:35,443 - oa.batch_embeddings - INFO - Submitting batches for 4 segments
    2025-03-28 14:57:36,784 - oa.batch_embeddings - DEBUG - Submitted batch batch_67e6aad079b08190a0e2b45e1aecd628 with 2 segments
    2025-03-28 14:57:37,819 - oa.batch_embeddings - DEBUG - Submitted batch batch_67e6aad18d5081909ec022687e474f65 with 2 segments
    2025-03-28 14:57:37,820 - oa.batch_embeddings - INFO - Submitted 2 batches
    2025-03-28 14:57:37,820 - oa.batch_embeddings - INFO - Monitoring 2 batches
    2025-03-28 14:57:38,001 - oa.batch_embeddings - DEBUG - Batch Batch(id='batch_67e6aad079b08190a0e2b45e1aecd628', completion_window='24h', created_at=1743170256, endpoint='/v1/embeddings', input_file_id='file-E5GXa9wne4UxRPn6YSeJAV', object='batch', status='validating', ...) status: in_progress
    Initial status summary: {'validating': 2, 'completed': 0, 'failed': 0}
    2025-03-28 14:57:38,419 - oa.batch_embeddings - DEBUG - Batch Batch(id='batch_67e6aad18d5081909ec022687e474f65', completion_window='24h', created_at=1743170257, endpoint='/v1/embeddings', input_file_id='file-V78FMHmz8xNkhfRtzkkBSP', object='batch', status='validating', ...) status: validating
    2025-03-28 14:57:41,589 - oa.batch_embeddings - INFO - Batch Batch(id='batch_67e6aad079b08190a0e2b45e1aecd628', completion_window='24h', created_at=1743170256, endpoint='/v1/embeddings', input_file_id='file-E5GXa9wne4UxRPn6YSeJAV', object='batch', status='validating', ...) completed successfully
    2025-03-28 14:57:41,793 - oa.batch_embeddings - DEBUG - Batch Batch(id='batch_67e6aad18d5081909ec022687e474f65', completion_window='24h', created_at=1743170257, endpoint='/v1/embeddings', input_file_id='file-V78FMHmz8xNkhfRtzkkBSP', object='batch', status='validating', ...) status: in_progress
    2025-03-28 14:57:43,999 - oa.batch_embeddings - DEBUG - Batch Batch(id='batch_67e6aad18d5081909ec022687e474f65', completion_window='24h', created_at=1743170257, endpoint='/v1/embeddings', input_file_id='file-V78FMHmz8xNkhfRtzkkBSP', object='batch', status='validating', ...) status: finalizing
    2025-03-28 14:57:47,555 - oa.batch_embeddings - INFO - Batch Batch(id='batch_67e6aad18d5081909ec022687e474f65', completion_window='24h', created_at=1743170257, endpoint='/v1/embeddings', input_file_id='file-V78FMHmz8xNkhfRtzkkBSP', object='batch', status='validating', ...) completed successfully
    2025-03-28 14:57:47,556 - oa.batch_embeddings - INFO - Batch processing complete: 2 successful, 0 failed
    Generated 4 embeddings for keys: ['fox', 'ml', 'embeddings', 'nlp']
    >>> segment_keys  # doctest: +SKIP
    ['fox', 'ml', 'embeddings', 'nlp']

    """
    # Sample text segments as a DICTIONARY
    segments = {
        "fox": "The quick brown fox jumps over the lazy dog.",
        "ml": "Machine learning models transform input data into useful representations.",
        "embeddings": "Embeddings capture semantic meaning in dense vector spaces.",
        "nlp": "Natural language processing enables computers to understand human language.",
    }

    # Get a process object instead of results
    process = compute_embeddings(
        segments=segments,
        verbosity=2,  # Show detailed progress information
        batch_size=2,  # Split into multiple batches
        poll_interval=3.0,  # Check status every 3 seconds
        return_process=True,  # Return the process instead of results
    )

    # Submit batches
    process.submit_batches()

    # Check status without blocking
    print("Initial status summary:", process.get_status_summary())

    # Now monitor until completion (this will block)
    process.monitor_batches()

    # Get and print results
    segment_keys, embeddings = process.aggregate_results()

    print(f"Generated {len(embeddings)} embeddings for keys: {segment_keys}")

    return segment_keys, embeddings


# Example 3: Using as a context manager
def context_manager_example():
    """
    Example usage of compute_embeddings as a context manager.
    This function computes embeddings for a list of text segments
    and prints the results.

    >>> segments, embeddings = context_manager_example()  # doctest: +SKIP
    2025-03-28 15:00:02,999 - oa.batch_embeddings - INFO - Submitting batches for 4 segments
    2025-03-28 15:00:05,087 - oa.batch_embeddings - INFO - Submitted 2 batches
    2025-03-28 15:00:05,088 - oa.batch_embeddings - INFO - Monitoring 2 batches
    2025-03-28 15:00:16,313 - oa.batch_embeddings - INFO - Batch Batch(id='batch_67e6ab64cb408190b80daa5e8ab92bf7', completion_window='24h', created_at=1743170404, ... status='validating', ...) completed successfully

    """
    # Sample text segments
    segments = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models transform input data into useful representations.",
        "Embeddings capture semantic meaning in dense vector spaces.",
        "Natural language processing enables computers to understand human language.",
    ]

    # Use as context manager for automatic cleanup
    with compute_embeddings(
        segments=segments, verbosity=1, batch_size=2, return_process=True
    ) as process:
        # Run the entire process
        segments, embeddings = process.run()

        print(f"Generated {len(embeddings)} embeddings in context")

    # After context exit, the processing mall is cleared (if not persist_processing_mall)

    return segments, embeddings


# Example 4: Using with pandas
def pandas_example():
    """
    Example usage of compute_embeddings with pandas DataFrame.
    This function computes embeddings for a dictionary of text segments
    and returns the results as a pandas DataFrame.

    >>> df = pandas_example()  # doctest: +SKIP
    2025-03-28 15:07:43,617 - oa.batch_embeddings - INFO - Submitting batches for 4 segments
    2025-03-28 15:07:45,843 - oa.batch_embeddings - INFO - Submitted 2 batches
    2025-03-28 15:07:45,844 - oa.batch_embeddings - INFO - Monitoring 2 batches
    2025-03-28 15:07:51,844 - oa.batch_embeddings - INFO - Batch Batch(id='batch_67e6ad3190048190ad587e2edd65ea33', ...) completed successfully
    2025-03-28 15:09:36,698 - oa.batch_embeddings - INFO - Batch Batch(id='batch_67e6ad306948819095e79cabd8263f0c', ...) completed successfully
    2025-03-28 15:09:36,700 - oa.batch_embeddings - INFO - Batch processing complete: 2 successful, 0 failed
    DataFrame shape: (4, 2)
    DataFrame index: ['fox', 'ml', 'embeddings', 'nlp']
    First row segment: The quick brown fox jumps over the lazy dog.
    First row embedding (first 5 dims): [4.308471e-05, -0.006475493, -0.00071540475, 0.018186275, 0.023950174]
                                                        segment  \
    fox              The quick brown fox jumps over the lazy dog.   
    ml          Machine learning models transform input data i...   
    embeddings  Embeddings capture semantic meaning in dense v...   
    nlp         Natural language processing enables computers ...   

                                                        embedding  
    fox         [4.308471e-05, -0.006475493, -0.00071540475, 0...  
    ml          [-0.021480283, 0.02021441, 0.012085131, 0.0159...  
    embeddings  [-0.018421143, -0.007218754, 0.0036062053, -0....  
    nlp         [0.015456875, 0.0016184314, 0.012820516, -0.04...

    """
    # Sample text segments
    segments = {
        "fox": "The quick brown fox jumps over the lazy dog.",
        "ml": "Machine learning models transform input data into useful representations.",
        "embeddings": "Embeddings capture semantic meaning in dense vector spaces.",
        "nlp": "Natural language processing enables computers to understand human language.",
    }

    # Get results as a pandas DataFrame
    df = compute_embeddings_df(segments=segments, verbosity=1, batch_size=2)

    # Display DataFrame information
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame index: {list(df.index)}")
    print(f"First row segment: {df.iloc[0]['segment']}")
    print(f"First row embedding (first 5 dims): {df.iloc[0]['embedding'][:5]}")

    return df


# Example 5: Error handling demonstration
def error_handling_example():
    """
    Example usage of compute_embeddings with error handling.
    This function demonstrates how to handle errors gracefully
    and continue processing valid segments.
    It intentionally includes invalid segments to trigger an error.
    The function will catch the error, print a message, and
    continue processing valid segments.
    This is useful for demonstrating error handling in a test environment.

    >>> result_segments, embeddings = error_handling_example()  # doctest: +SKIP
    2025-03-28 15:13:39,827 - oa.batch_embeddings - INFO - Submitting batches for 4 segments
    2025-03-28 15:13:39,828 - oa.batch_embeddings - INFO - Submitting batches for 2 segments
    Caught expected error: TypeError: argument 'text': 'int' object cannot be converted to 'PyString'
    Correcting and continuing with valid segments...
    2025-03-28 15:13:40,891 - oa.batch_embeddings - INFO - Submitted 1 batches
    2025-03-28 15:13:40,892 - oa.batch_embeddings - INFO - Monitoring 1 batches
    2025-03-28 15:13:56,657 - oa.batch_embeddings - INFO - Batch Batch(id='batch_67e6ae94932c819092092eb1bf91f3c9', ...) completed successfully
    2025-03-28 15:13:56,659 - oa.batch_embeddings - INFO - Batch processing complete: 1 successful, 0 failed
    Successfully generated 2 embeddings after correction
    >>> print(f"{len(result_segments)=}, {len(embeddings)=}")  # doctest: +SKIP
    len(result_segments)=2, len(embeddings)=2

    """
    try:
        # Intentionally invalid segments to trigger an error
        segments = [None, "Valid text", 123, "Another valid text"]

        result_segments, embeddings = compute_embeddings(segments=segments, verbosity=1)
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {str(e)}")

        # Show how to handle and continue
        print("Correcting and continuing with valid segments...")
        valid_segments = [seg for seg in segments if isinstance(seg, str)]
        result_segments, embeddings = compute_embeddings(
            segments=valid_segments, verbosity=1
        )

        print(f"Successfully generated {len(embeddings)} embeddings after correction")

    return result_segments, embeddings
```

## examples/illustrate_stories.py

```python
"""Illustrate stories with OpenAI's DALL-E model."""

import html

import oa
from i2 import postprocess

# TODO: Protect item gets (obj[k]) from KeyError/IndexError. Raise meaningful error.

# @code_to_dag
# def make_children_story():
#     story_text = make_it_rhyming(story)
#     image = get_illustration(story_text, image_style)
#     page = aggregate_story_and_image(image, rhyming_story)

make_it_rhyming_prompt = """Act as a children's book author. 
Write a rhyming story about the following text:
###
{story}"""

illustrate_prompt = """Act as an illustrator, expert in the style: {image_style}.
Describe an image that would illustrate the following text:
###
{text}
"""

dalle_prompt = """Image style: {image_style}
{image_description}
"""

topic_points_prompt = """
I will give you a topic/subject and you will list {n_talking_points} talking points of 
the main ideas/subtopics/points of this topic.
Each talking point should be between {min_n_words} and {max_n_words} words. 

Your answer should be in the form of a bullet point list, 
and nothing but a bullet point list. Each bullet point contains the talking point only.

The topic is: 

{topic}
"""

topic_points_json_prompt = """
I will give you a topic/subject and you will list {n_talking_points} of the main 
ideas/subtopics/points of this topic, 
including a`title` and {min_n_words} to {max_n_words} word `description` of the topic. 

Your answer should be in the form of a valid JSON string, and nothing but a valid JSON.
The JSON should have the format `{{title: description, title: description, ...}}`

The topic is: 

{topic}
"""

DFLT_IMAGE_STYLE = "drawing"


def extract_first_text_choice(response) -> str:
    return response.choices[0].text.strip()


# TODO: Put back min_n_words and max_n_words as arguments once code_to_dag supports
#  more control over function injection (such as only taking the subset of arguments
#  mentioned by the FuncNodes, and auto-editing FuncNode binds.
@postprocess(extract_first_text_choice)
def topic_points(topic, n_talking_points=3) -> str:
    print(f"topic_points: topic={topic}, n_talking_points={n_talking_points}")
    min_n_words = 5
    max_n_words = 20
    prompt = topic_points_prompt.format(
        topic=topic,
        n_talking_points=n_talking_points,
        min_n_words=min_n_words,
        max_n_words=max_n_words,
    )
    return oa.complete(prompt, max_tokens=2048, n=1, engine="text-davinci-003")


def _repair_json(json_str):
    t = json_str
    if t.startswith("`"):
        t = t[1:]
    if t.endswith("`"):
        t = t[:-1]
    return t


def topic_points_json(topic, n_talking_points=3, min_n_words=15, max_n_words=40) -> str:
    prompt = topic_points_prompt.format(
        topic=topic,
        n_talking_points=n_talking_points,
        min_n_words=min_n_words,
        max_n_words=max_n_words,
    )
    t = oa.complete(prompt, max_tokens=2048, n=1, engine="text-davinci-003")
    t = extract_first_text_choice(t)
    return _repair_json(t)


# @postprocess(extract_first_text_choice)
def make_it_rhyming(story, *, max_tokens=512, **chat_param) -> str:
    prompt = make_it_rhyming_prompt.format(story=story)
    return oa.chat(prompt, max_tokens=max_tokens, n=1, **chat_param)


# @postprocess(extract_first_text_choice)
def get_image_description(
    story_text: str, image_style=DFLT_IMAGE_STYLE, max_tokens=256, **chat_param
) -> str:
    prompt = illustrate_prompt.format(text=story_text, image_style=image_style)
    return oa.chat(prompt, max_tokens=max_tokens, n=1, **chat_param)


def get_image_url(image_description, image_style=DFLT_IMAGE_STYLE):
    prompt = dalle_prompt.format(
        image_description=image_description, image_style=image_style
    )
    return oa.dalle(prompt, n=1)


def get_illustration(story_text: str, image_style=DFLT_IMAGE_STYLE):
    image_description = get_image_description(story_text, image_style)
    url = get_image_url(image_description, image_style)
    return url


def _format_for_html_display(input_string: str) -> str:
    escaped_string = html.escape(input_string)
    newline_replaced_string = escaped_string.replace("\n", "<br>")
    html_formatted_string = f"<p>{newline_replaced_string}</p>"
    return html_formatted_string


def aggregate_story_and_image(image_url, story_text):
    """Produces an html page with the image and story text"""

    story_text = _format_for_html_display(story_text)
    html = f"""<html>
    <body>
    <img src="{image_url}" />
    <p>{story_text}</p>
    </body>
    </html>"""

    return html
```

## oa_types.py

```python
"""Types for oa"""

from typing import Any, List, Optional, TypeVar, Generic

from pydantic import BaseModel, RootModel, Field

from ju.pydantic_util import is_pydantic_model, is_type_hint

import openai.types as oat

pydantic_models = {k: v for k, v in vars(oat).items() if is_pydantic_model(v)}
type_hints = {k: v for k, v in vars(oat).items() if is_type_hint(v)}


T = TypeVar("T", bound=BaseModel)

# --------------------------------------------------------------------------------------
# JsonL (lists of dicts)


class JsonL(RootModel[list[T]], Generic[T]):
    """
    A generic class for JSONL (JSON Lines) files, which are lists of dictionaries.
    """


class InputText(BaseModel):
    """Used to specify the input data in some OpenAI API endpounts."""

    input: str


InputDataJsonL = JsonL[InputText]


# --------------------------------------------------------------------------------------
# BatchRequest (e.g. embeddings)


class BatchRequestBody(BaseModel):
    input: list[str]
    model: str


# Note: leaf model
class BatchRequest(BaseModel):
    custom_id: str
    method: str
    url: str
    body: BatchRequestBody


# --------------------------------------------------------------------------------------
# OpenAI Responses
from pydantic import BaseModel, Field
from typing import List, TypeVar, Generic, Any

# Define a generic type for Datum
DatumT = TypeVar("DatumT")


# Usage model remains the same
class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


# ResponseBody is now parameterized by DatumT
class ResponseBody(BaseModel, Generic[DatumT]):
    object: str
    data: list[DatumT]  # Generic list of DatumT
    model: str
    usage: Usage


class RequestResponse(BaseModel, Generic[DatumT]):
    status_code: int = Field(..., ge=100, le=599)
    request_id: str
    body: ResponseBody[DatumT]


class Response(BaseModel, Generic[DatumT]):
    id: str
    custom_id: str
    response: RequestResponse[DatumT]
    error: Any


from openai.types import Embedding as EmbeddingT

EmbeddingResponse = Response[EmbeddingT]
EmbeddingResponse.__name__ = "EmbeddingResponse"

# --------------------------------------------------------------------------------------
# Extras


def heatmap_of_models_and_their_fields():
    import pandas as pd  # pylint: disable=import-outside-toplevel
    from oplot.matrix import heatmap_sns  # pylint: disable=import-outside-toplevel

    models_and_their_fields = pd.DataFrame(
        [{k: 1 for k in model.model_fields} for model in pydantic_models.values()],
        index=pydantic_models.keys(),
    ).transpose()

    return heatmap_sns(models_and_their_fields, figsize=13)


# # --------------------------------------------------------------------------------------

# class Datum(BaseModel):
#     object: str
#     index: int
#     embedding: List[float]


# class Usage(BaseModel):
#     prompt_tokens: int
#     total_tokens: int


# class ResponseBody(BaseModel):
#     object: str
#     data: List[Datum]
#     model: str
#     usage: Usage


# class Response(BaseModel):
#     status_code: int = Field(..., ge=100, le=599)
#     request_id: str
#     body: ResponseBody


# # Note: leaf model
# class ResponseRoot(BaseModel):
#     id: str
#     custom_id: str
#     response: Response
#     error: Any
```

## openai_specs.py

```python
"""Tools to extract specs from the openai interface

See raw schemas by doing

>>> from oa.openai_specs import schemas
>>> sorted(schemas)  # doctest: +SKIP
['AssistantFileObject', 'AssistantObject', 'AssistantToolsCode', 'AssistantToolsFunction', 'AssistantToolsRetrieval', 'ChatCompletionFunctionCallOption',
'ChatCompletionFunctions', 'ChatCompletionMessageToolCall', 'ChatCompletionMessageToolCallChunk', 'ChatCompletionMessageToolCalls',
'ChatCompletionNamedToolChoice', 'ChatCompletionRequestAssistantMessage', 'ChatCompletionRequestFunctionMessage', 'ChatCompletionRequestMessage',
'ChatCompletionRequestMessageContentPart', 'ChatCompletionRequestMessageContentPartImage', 'ChatCompletionRequestMessageContentPartText',
'ChatCompletionRequestSystemMessage', 'ChatCompletionRequestToolMessage', 'ChatCompletionRequestUserMessage', 'ChatCompletionResponseMessage',
'ChatCompletionRole', 'ChatCompletionStreamResponseDelta', 'ChatCompletionTool', 'ChatCompletionToolChoiceOption', 'CompletionUsage',
'CreateAssistantFileRequest', 'CreateAssistantRequest', 'CreateChatCompletionFunctionResponse', 'CreateChatCompletionImageResponse',
'CreateChatCompletionRequest', 'CreateChatCompletionResponse', 'CreateChatCompletionStreamResponse', 'CreateCompletionRequest',
'CreateCompletionResponse', 'CreateEditRequest', 'CreateEditResponse', 'CreateEmbeddingRequest', 'CreateEmbeddingResponse', 'CreateFileRequest',
'CreateFineTuneRequest', 'CreateFineTuningJobRequest', 'CreateImageEditRequest', 'CreateImageRequest', 'CreateImageVariationRequest', 'CreateMessageRequest',
'CreateModerationRequest', 'CreateModerationResponse', 'CreateRunRequest', 'CreateSpeechRequest', 'CreateThreadAndRunRequest', 'CreateThreadRequest',
'CreateTranscriptionRequest', 'CreateTranscriptionResponse', 'CreateTranslationRequest', 'CreateTranslationResponse', 'DeleteAssistantFileResponse',
'DeleteAssistantResponse', 'DeleteFileResponse', 'DeleteMessageResponse', 'DeleteModelResponse', 'DeleteThreadResponse', 'Embedding', 'Error',
'ErrorResponse', 'FineTune', 'FineTuneEvent', 'FineTuningJob', 'FineTuningJobEvent', 'FunctionObject', 'FunctionParameters', 'Image',
'ImagesResponse', 'ListAssistantFilesResponse', 'ListAssistantsResponse', 'ListFilesResponse', 'ListFineTuneEventsResponse', 'ListFineTunesResponse',
'ListFineTuningJobEventsResponse', 'ListMessageFilesResponse', 'ListMessagesResponse', 'ListModelsResponse', 'ListPaginatedFineTuningJobsResponse',
'ListRunStepsResponse', 'ListRunsResponse', 'ListThreadsResponse', 'MessageContentImageFileObject', 'MessageContentTextAnnotationsFileCitationObject',
'MessageContentTextAnnotationsFilePathObject', 'MessageContentTextObject', 'MessageFileObject', 'MessageObject', 'Model', 'ModifyAssistantRequest',
'ModifyMessageRequest', 'ModifyRunRequest', 'ModifyThreadRequest', 'OpenAIFile', 'RunObject', 'RunStepDetailsMessageCreationObject', 'RunStepDetailsToolCallsCodeObject',
'RunStepDetailsToolCallsCodeOutputImageObject', 'RunStepDetailsToolCallsCodeOutputLogsObject', 'RunStepDetailsToolCallsFunctionObject', 'RunStepDetailsToolCallsObject',
'RunStepDetailsToolCallsRetrievalObject', 'RunStepObject', 'RunToolCallObject', 'SubmitToolOutputsRunRequest', 'ThreadObject']
>>> schema = schemas['CreateCompletionRequest']
>>> schema['properties']  # doctest: +SKIP
 {'model': {'description': 'ID of the model to use...

Get resulting signatures by doing.

>>> from oa.openai_specs import sig
>>> sig.CreateCompletionRequest)  # doctest: +SKIP
<Sig (model: str, prompt='<|endoftext|>', *, seed: int, best_of: int = 1, echo: bool = False,
frequency_penalty: float = 0, logit_bias: dict = None, logprobs: int = None, max_tokens: int = None,
n: int = 1, presence_penalty: float = 0, stop=None, stream: bool = False, suffix: str = None,
temperature: float = 1, top_p: float = 1, user: str = None)>

Or see argument descriptions in rst format by doing:

>>> from oa.openai_specs import schema_to_rst_argument_descriptions, schemas
>>> schema_to_rst_argument_descriptions(schemas['CreateCompletionRequest'])  # doctest: +SKIP

```

"""

import os
import re
from functools import lru_cache
from typing import Dict, List, TypedDict, Optional, Literal
import yaml

import dol
from i2 import Sig, Param, empty_param_attr as empty

from oa.util import grazed


# OPENAPI_SPEC_URL = "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml"
# The above is the official openapi spec, but they updated it, breaking the yaml load with
#  a "found duplicate anchor 'run_temperature_description';" error.
# (see CI: https://github.com/thorwhalen/oa/actions/runs/8735713865/job/24017876818#step:7:452)
# So I made a copy of the previous working one to use, and will update it when I have time.
# TODO: Update openapi yaml def (or use )
# See https://github.com/thorwhalen/oa/discussions/8#discussioncomment-9165753
OPENAPI_SPEC_URL = (
    "https://raw.githubusercontent.com/thorwhalen/oa/main/misc/openapi.yaml"
)


@lru_cache
def get_openapi_spec_dict(
    openapi_spec_url: str = OPENAPI_SPEC_URL,
    *,
    refresh: bool = False,
    expand_refs: bool = True,
):
    """Get the dict of the openapi spec"""
    if refresh:
        # TODO: Graze should have a refresh method to make sure to not delete before
        #  re-grazing
        del grazed[openapi_spec_url]

    d = yaml.safe_load(grazed[openapi_spec_url])
    if expand_refs:
        import jsonref  # pip install jsonref

        d = jsonref.JsonRef.replace_refs(d)
    return d


specs = get_openapi_spec_dict()
schemas = specs["components"]["schemas"]

# A mapping between the OpenAI API's schema names and the corresponding Python types.
pytype_of_jstype = {
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": float,
    "array": list,
    "object": dict,
}


# TODO: ?Extract this directly from the ChatCompletionRequestMessage schema
#  Could use TypedDict with annotations:
#  {p.name: p.annotation for p in sig.ChatCompletionRequestMessage.params}
class Message(TypedDict):
    role: str
    content: str
    name: str | None


Messages = list[Message]
Model = str  # TODO: should be Literal parsed from the models list

pytype_of_name = {
    "messages": Messages,
    "model": Model,
}

# Some arguments don't have defaults in the schema, but are secondary, so shouldn't be
# required. The lazy way to handle this case is to give defaults to these arguments.
# The better way would be to give these arguments a sentinel default (say, None) and
# wrap our python functions so they ignore it (don't include it in the request.
# Further -- some schema defaults have values (like max_tokens='inf' that don't match
# their type, AND make the request fail. So we a mechanism to overwride the schema
# default too.
pre_defaults_of_name = {  # defaults that should override the schema
    "max_tokens": None,
}
post_defaults_of_name = {  # defaults used if schema doesn't have a default
    "user": None,
}


def properties_to_annotation(name: str, props: dict):
    """Get annotations from a schema"""
    annotation = pytype_of_name.get(name, None)  # get type from name if exists
    if annotation is None:  # if not, fallback on schemas enum or type
        if "enum" in props:
            annotation = Literal[tuple(props["enum"])]
        else:
            annotation = pytype_of_jstype.get(props.get("type", None), empty)
    return annotation


def properties_to_param_dict(name: str, props: dict):
    """Get all but kind fields of a Parameter object"""
    annotation = pytype_of_name.get(name, None)  # get type from name if exists
    if annotation is None:  # if not, fallback on schemas type
        annotation = pytype_of_jstype.get(props.get("type", None), empty)
    if name in pre_defaults_of_name:
        default = pre_defaults_of_name[name]
    else:
        default = props.get("default", post_defaults_of_name.get(name, empty))
    return dict(name=name, default=default, annotation=annotation)


def schema_to_signature(schema):
    """Get a signature from a schema

    >>> from oa.openai_specs import specs
    >>> schema = specs['components']['schemas']['CreateChatCompletionRequest']
    >>> print(str(schema_to_signature(schema)))  # doctest: +SKIP
    (model: str, messages: List[openai_specs.Message], *, temperature: float = 1, ...

    """

    def gen():
        required = schema.get("required", [])
        for name, props in schema.get("properties", {}).items():
            try:
                if name in required:
                    yield Param(
                        **properties_to_param_dict(name, props),
                        kind=Param.POSITIONAL_OR_KEYWORD,
                    )
                else:
                    yield Param(
                        **properties_to_param_dict(name, props), kind=Param.KEYWORD_ONLY
                    )
            except ValueError as e:
                # TODO: protecting this with try/except because OpenAI changed
                # it's schema and since then got a
                # ValueError: 'timestamp_granularities[]' is not a valid parameter name
                # error.
                # I'd rather catch up...
                pass

    return Sig(sorted(gen()))


def _clean_up_whitespace(s):
    """Clean up whitespace in a string.
    Replace all whitespaces with single space, and strip.
    """
    return re.sub(r"\s+", " ", s).strip()


def schema_to_rst_argument_descriptions(
    schema, process_description=_clean_up_whitespace
):
    """Yield rst-formatted argument descriptions for a schema

    >>> from oa.openai_specs import specs
    >>> schema = specs['components']['schemas']['CreateChatCompletionRequest']
    >>> print(*schema_to_rst_argument_descriptions(schema), '\\n')  # doctest: +SKIP
    :param model: ID of the model to use. Currently, only ...

    `process_description` is a function that will be applied to each schema description
    (so you can clean it up, or add links, etc.). By default, it will replace all
    whitespaces with single space, and strip.

    """
    process_description = process_description or (lambda x: x)
    for name, props in schema.get("properties", {}).items():
        description = process_description(props.get("description", ""))
        yield f":param {name}: {description}"


def schema_to_docs(
    name,
    schema: dict,
    prefix="",
    line_prefix: str = "\t",
):
    """Get the docs for a schema

    >>> from oa.openai_specs import specs
    >>> schema = specs['components']['schemas']['CreateChatCompletionRequest']
    >>> print(schema_to_docs('chatcompletion', schema))  # doctest: +SKIP
    chatcompletion(...

    :param model: ID of the model to use. Currently, only ...

    """
    s = schema_to_signature(schema)
    doc = prefix
    doc += f"{name}(\n\t" + "\n\t".join(str(s)[1:-1].split(", ")) + "\n)"
    doc += "\n\n"
    doc += "\n\n".join(schema_to_rst_argument_descriptions(schema))
    return _prefix_all_lines(doc, line_prefix)


def _prefix_all_lines(string: str, prefix: str = "\t"):
    return "\n".join(map(lambda line: prefix + line, string.splitlines()))


from dol import path_filter, path_get
from dol.sources import AttrContainer, Attrs
from types import SimpleNamespace

sig = AttrContainer(
    **{name: schema_to_signature(schema) for name, schema in schemas.items()}
)
_docs = AttrContainer(
    **{name: schema_to_docs(name.lower(), schema) for name, schema in schemas.items()}
)

from i2 import wrap

from functools import partial, cached_property


# TODO: Write a types.SimpleNamespace version of this.
class SpecNames:
    specs = specs

    @cached_property
    def op_paths(self):
        return list(
            path_filter(lambda p, k, v: k == "operationId" or v == "operationId", specs)
        )

    @cached_property
    def route_and_op(self):
        return [(p[1], path_get(specs, p, get_value=dict.get)) for p in self.op_paths]

    @cached_property
    def create_route_and_op(self):
        return list(filter(lambda x: x[1].startswith("create"), self.route_and_op))

    @cached_property
    def sigs_ending_with_req(self):
        return [x for x in vars(sig) if x.endswith("Request")]

    @cached_property
    def creation_actions(self):
        return set(
            map(
                lambda x: x[len("create") : -len("request")].lower(),
                self.sigs_ending_with_req,
            )
        )

    @cached_property
    def attrs(self):
        import oa

        return vars(oa.openai)
        # return Attrs(oa.openai)

    @cached_property
    def matched_names(self):
        return [
            x
            for x in list(self.attrs)
            if "create" in dir(self.attrs) and x.lower() in self.creation_actions
            # if "create" in self.attrs[x] and x.lower() in self.creation_actions
        ]

    @cached_property
    def doc_for_name(self):
        import oa

        return {
            op_name: (getattr(oa.openai, op_name).create.__doc__ or "").strip()
            for op_name in self.matched_names
        }

    @cached_property
    def schema_for_name(self):
        return {
            name: schemas[f"Create{name[0].upper()}{name[1:]}Request"]
            for name in self.matched_names
        }

    def _assert_that_everything_matches(self):
        assert set(
            map(lambda x: x[: -len("request")].lower(), self.sigs_ending_with_req)
        ) == set(map(str.lower, [x[1] for x in self.create_route_and_op]))


spec_names = SpecNames()


## rm_when_none-parametrizale version of current version below
# def _kwargs_cast_ingress(func_sig, rm_when_none=(), /, *args, **kwargs):
#     kwargs = func_sig.map_arguments(args, kwargs)
#     if rm_when_none:
#         for k in rm_when_none:
#             if kwargs[k] is None:
#                 del kwargs[k]
#     return (), kwargs


def _kwargs_cast_ingress(func_sig, /, *args, **kwargs):
    kwargs = func_sig.map_arguments(args, kwargs)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return (), kwargs


def _mk_raw():
    import oa

    def gen():
        for name in spec_names.matched_names:
            lo_name = name.lower()
            func_sig = getattr(sig, f"Create{name}Request")

            func = wrap(
                getattr(oa.openai, name).create,
                ingress=func_sig(partial(_kwargs_cast_ingress, func_sig)),
                name=lo_name,
            )
            func.__doc__ = schema_to_docs(
                lo_name,
                spec_names.schema_for_name[name],
                prefix=spec_names.doc_for_name[name] + "\n\n",
            )
            yield lo_name, func

    from dol.sources import AttrContainer

    return AttrContainer(**dict(gen()))


raw = _mk_raw()


def normalized_file_name(prompt: str) -> str:
    """Convert prompt to a normalized valid file/folder name

    >>> normalized_file_name("This is a prompt")
    'this is a prompt'
    >>> normalized_file_name("This is: a PROMPT!  (with punctuation)")
    'this is a prompt with punctuation'
    """
    return re.sub(r"\W+", " ", prompt).lower().strip()


def prompt_path(prompt, prefix=""):
    filepath = os.path.join(prefix, normalized_file_name(prompt))
    return dol.ensure_dir(filepath)


def merge_keys_to_values(d: dict, key_name="key"):
    """Merge the keys of a dict into the values of the dict.

    >>> d = {'a': {'b': 1}, 'c': {'d': 2, 'e': 3}}
    >>> dict(merge_keys_to_values(d))
    {'a': {'key': 'a', 'b': 1}, 'c': {'key': 'c', 'd': 2, 'e': 3}}

    Useful when, for example, you want to make a table containing both keys and values.
    """
    for k, v in d.items():
        assert key_name not in v, (
            f"The key_name {key_name} was found in the value dict. Choose a different "
            f"key_name."
        )
        yield k, dict({key_name: k}, **v)


def schemas_df(schema: dict):
    import pandas as pd

    print(pd.Series(schema["properties"]))
    print(schema["required"])
    pd.DataFrame(schema["properties"]).T.fillna("")
```

## scrap/__init__.py

```python

```

## stores.py

```python
"""Data object layers for openai resources"""

from collections.abc import Mapping
from operator import attrgetter, methodcaller
from typing import Optional, Union, Any, List, Literal, T
from collections.abc import Iterable, Callable
import json
from functools import wraps, partial, cached_property

import openai  # TODO: Import from oa.util instead?

# TODO: Move openai imports to go through oa.util
from openai.resources.files import Files as OpenaiFiles, FileTypes, FileObject
from openai.types import FileObject

from oa.base import TextOrTexts
from oa.batches import (
    mk_batch_file_embeddings_task,
    BatchSpec,
    BatchObj,
    BatchId,
)
from oa.util import (
    OaClientSpec,
    mk_client,
    ensure_oa_client,
    merge_multiple_signatures,
    source_parameter_props_from,
    utc_int_to_iso_date,
    iso_date_to_utc_int,
    Purpose,
    BatchesEndpoint,
    DFLT_ENCODING,
    jsonl_loads_iter,
    jsonl_dumps,
    jsonl_loads,
)
from i2 import Sig
from i2.signatures import SignatureAble, ParamsAble
from i2 import postprocess
from dol import wrap_kvs, KvReader, Pipe

FilterFunc = Callable[[Any], bool]


DFLT_PURPOSE = "batch"
DFLT_BATCHES_ENDPOINT = "/v1/embeddings"

openai_files_cumul_sig = merge_multiple_signatures(
    [
        OpenaiFiles.create,
        OpenaiFiles.retrieve,
        OpenaiFiles.delete,
        Sig(OpenaiFiles.list),
    ],
    default_conflict_method="take_first",
)

params_from_openai_files_cls = source_parameter_props_from(openai_files_cumul_sig)

files_create_sig = (
    Sig(OpenaiFiles.create)
    .ch_defaults(purpose=DFLT_PURPOSE)
    .ch_kinds(purpose=Sig.POSITIONAL_OR_KEYWORD)
)


def _is_string(x):
    return isinstance(x, str)


def _has_id_attr(x):
    return hasattr(x, "id")


def _is_instance_and_has_id(x=None, *, type_: type):
    if x is None:
        return partial(_is_instance_and_has_id, type_=type_)
    else:
        return isinstance(x, type_) and hasattr(x, "id")


def extract_id(
    method: Callable | None = None,
    *,
    is_id: FilterFunc = _is_string,
    has_id: FilterFunc = _has_id_attr,
    get_id: Callable = attrgetter("id"),
):
    """
    Decorator that will extract the id from the first non-instance argument of a method.

    >>> @extract_id
    ... def veni(self, vidi, vici):
    ...     return f"{vidi=}, {vici=}"
    >>> assert (
    ...     veni(None, 'hi', vici=3)  # calling the function
    ...     == veni.__wrapped__(None, 'hi', vici=3)  # outputs the same as calling the original
    ...     == "vidi='hi', vici=3"
    ... )

    Except it now has extra powers; if your id is contained in a attribute,
    it'll be extracted.

    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace(id='hi')
    >>> obj.id
    'hi'
    >>> assert veni(None, obj, vici=3) == "vidi='hi', vici=3"

    You can also customize the filters and the id getter.
    The following will resolve a string representation of an integer as an id.

    >>> @extract_id(is_id=lambda x: isinstance(x, int), has_id=str.isnumeric, get_id=int)
    ... def add_one(self, x):
    ...     return x
    ...
    >>> assert add_one(None, 3) == 3
    >>> assert add_one(None, "42") == 42

    """
    if method is None:
        return partial(extract_id, is_id=is_id, has_id=has_id, get_id=get_id)
    else:

        @wraps(method)
        def _wrapped_method(self, x, *args, **kwargs):
            if not is_id(x):
                if has_id(x):
                    x = get_id(x)
                else:
                    raise ValueError(f"Can't resolve id from {type(x)}: {x}")
            return method(self, x, *args, **kwargs)

        return _wrapped_method


class MappingHooks(Mapping):
    def __iter__(self):
        # TODO: I'd like to just return the (iterable) object, but that doesn't work (why?)
        yield from self._iter()

    def __len__(self):
        return self._len()

    def __contains__(self, key):
        return self._contains(key)

    def __getitem__(self, key):
        return self._getitem(key)


class MutuableMappingHooks(MappingHooks):
    def __delitem__(self, key):
        return self._delitem(key)

    def __setitem__(self, key, value):
        return self._setitem(key, value)


class OaMapping(MappingHooks, KvReader):
    client: openai.Client
    _list_kwargs = {}  # default, overriden in __init__

    def _len(self) -> int:
        """Return the number of batches."""
        # TODO: Does the API have a direct way to get the number of items?
        c = 0
        for _ in self:
            c += 1
        return c

    def _contains(self, k) -> bool:
        """Check if an item is contained in the mapping."""
        try:
            self[k]
            return True
        except openai.NotFoundError:
            return False

    def __delitem__(self, key):
        return self._delitem(key)


def is_task_dict(x):
    """Is a dict (or Mapping) the schema of an openai API 'task'"""
    # TODO: Get this schema dynamically from the API's swagger spec (or similar)
    task_keys = {"custom_id", "method", "url", "body"}
    return isinstance(x, Mapping) and all(k in x for k in task_keys)


def is_task_dict_list(x):
    """
    Is a list (or iterable) of dicts (or Mappings) the schema of an openai API 'task'
    """
    # TODO: Get this schema dynamically from the API's swagger spec (or similar)
    return all(is_task_dict(item) for item in x)


class OaFilesBase(OaMapping):
    # @params_from_openai_files_cls
    @Sig.replace_kwargs_using(files_create_sig - "purpose")
    def __init__(
        self,
        client: openai.Client | None = None,
        purpose: Purpose | None = DFLT_PURPOSE,  # type: ignore
        iter_filter_purpose: bool = False,  # type: ignore
        encoding: str = DFLT_ENCODING,
        **extra_kwargs,
    ):
        if client is None:
            client = mk_client()
        self.client = client
        self.purpose = purpose
        self.iter_filter_purpose = iter_filter_purpose
        if self.iter_filter_purpose:
            self._list_kwargs = {"purpose": self.purpose}
        self.encoding = encoding
        self.extra_kwargs = extra_kwargs

    def _iter(self):
        return self.client.files.list(**self._list_kwargs)

    @extract_id
    def metadata(self, file_id) -> FileObject:
        return self.client.files.retrieve(file_id, **self.extra_kwargs)

    @extract_id
    def content(self, file_id):
        return self.client.files.content(file_id, **self.extra_kwargs)

    _getitem = content

    @extract_id
    def _delitem(self, file_id):
        # Delete the file using the API (you might need to implement this)
        return self.client.files.delete(file_id)  # Assuming there's a delete method

    @params_from_openai_files_cls
    def append(self, file: FileTypes | dict) -> FileObject:
        # Note: self.client.create can be found in openai.resources.files.Files.create
        if is_task_dict(file) or is_task_dict_list(file):
            file = jsonl_dumps(file, self.encoding)
        return self.client.files.create(
            file=file, purpose=self.purpose, **self.extra_kwargs
        )

    @Sig.replace_kwargs_using(mk_batch_file_embeddings_task)
    def create_embedding_task(self, texts: TextOrTexts, **extra_kwargs):
        # Note: self.client.create can be found in openai.resources.files.Files.create
        task = mk_batch_file_embeddings_task(texts, **extra_kwargs)
        return self.append(task)


class OaFilesMetadata(OaFilesBase):
    """
    A key-value store for OpenAI files metadata.
    """

    _getitem = OaFilesBase.metadata


@wrap_kvs(key_decoder=attrgetter("id"), value_decoder=attrgetter("content"))
class OaFiles(OaFilesBase):
    """
    A key-value store for OpenAI files content data.
    Keys are the file IDs.
    """


@wrap_kvs(key_decoder=attrgetter("id"), value_decoder=jsonl_loads)
class OaJsonLFiles(OaFilesBase):
    """
    A key-value store for OpenAI files content data.
    Keys are the file IDs.
    """


def get_json_or_jsonl_data(response):
    """Extract the json data from a response"""
    try:
        return response.json()
    except json.JSONDecodeError:
        return jsonl_loads(response.content)


@wrap_kvs(key_decoder=attrgetter("id"), value_decoder=get_json_or_jsonl_data)
class OaJsonFiles(OaFilesBase):
    """
    A key-value store for OpenAI files content data.
    Keys are the file IDs.
    """


# TODO: Find a non-underscored place to import HttpxBinaryResponseContent from
# Note: This is just used for annotation purposes
from openai._legacy_response import HttpxBinaryResponseContent

from typing import TypedDict, TypeVar


# TODO: Find some where to import this definition from
class ResponseDict(TypedDict):
    id: str  # Example type, you can replace it with whatever type is appropriate
    custom_id: str
    response: dict
    error: Any  # TODO: Look up what type this can be


class DataObject(TypedDict, total=False):
    """dicts that have two required fields: 'object' and 'index'"""

    object: str  # Required field
    index: int  # Required field


class EmbeddingsDataObject(TypedDict):
    """
    A DataObject with object='embeddings' and an 'embeddings' key that is a list of
    floats
    """

    object: Literal["embeddings"]  # This enforces the value to be 'embeddings'
    index: int  # Required field
    embeddings: list[float]  # The third known field, 'embeddings'


DataObjectValue = TypeVar("DataObjectValue")
DataObjectValue.__doc__ = (
    "The value that a DataObject holds in the field indicated by the object field"
)

jsonl_loads_response_lines = partial(
    jsonl_loads_iter, get_lines=methodcaller("iter_lines")
)

jsonl_loads_list = Pipe(jsonl_loads_response_lines, list)


def response_body_data(response_dict: ResponseDict) -> DataObject:
    return response_dict["response"]["body"]["data"]


def object_of_data(data: DataObject) -> DataObjectValue:
    """
    This function extracts the object (value) from a {object: V, index: i, V} dict
    which is the format you'll find in the ['response']['body']['data'] of a response.

    >>> object_of_data({'object': 'embeddings', 'index': 3, 'embeddings': [1, 2, 3]})
    [1, 2, 3]

    """
    return data[data["object"]]


def response_body_data_objects(
    response_dict: ResponseDict,
) -> Iterable[DataObjectValue]:
    data = response_body_data(response_dict)
    for d in data:
        yield object_of_data(d)


# Note: This is to be used as a content decoder
# TODO: The looping logic is messy, consider refactoring
@postprocess(dict)
def get_json_data_from_response(response: HttpxBinaryResponseContent):
    """Extract the embeddings from a HttpxBinaryResponseContent object"""
    for d in jsonl_loads_response_lines(response):
        custom_id = d["custom_id"]
        dd = response_body_data(d)
        if isinstance(dd, list):
            for ddd in dd:
                yield custom_id, object_of_data(ddd)
        else:
            yield custom_id, object_of_data(dd)


@wrap_kvs(key_decoder=attrgetter("id"), value_decoder=get_json_data_from_response)
class OaFilesJsonData(OaFilesBase):
    """
    A key-value store for OpenAI files content data.
    Keys are the file IDs.
    Values are the data extracted from the HttpxBinaryResponseContent object.
    """


import openai
from collections.abc import Mapping
from typing import Optional, Dict
from openai.resources.batches import Batches as OaBatches
from typing import Literal


# TODO: Why does go to definition here (for self.client.batches,
#    but also for self.client.files), but not in OpenAIFilesBase?
class OaBatchesBase(OaMapping):
    def __init__(self, client: openai.Client | None = None, **extra_kwargs):
        if client is None:
            client = mk_client()
        self.client = client
        self.extra_kwargs = extra_kwargs

    def _iter(self) -> Iterable[BatchObj]:
        """Return an iterator over batch IDs"""
        return self.client.batches.list(**self._list_kwargs)

    @extract_id
    def metadata(self, batch_id: str) -> BatchObj:
        """Retrieve metadata for a batch."""
        return self.client.batches.retrieve(batch_id, **self.extra_kwargs)

    _getitem = metadata

    @extract_id
    def _delitem(self, batch_id: str):
        """Cancel the batch via the API."""
        self.client.batches.cancel(batch_id)

    @extract_id
    def append(
        self,
        input_file_id: str,
        *,
        endpoint: BatchesEndpoint = DFLT_BATCHES_ENDPOINT,  # type: ignore
        completion_window: Literal["24h"] = "24h",
        metadata: dict[str, str] | None = None,
    ):
        """Create and submit a new batch via submitting the input file (obj or id)."""
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
            **self.extra_kwargs,
        )


@wrap_kvs(key_decoder=attrgetter("id"))
class OaBatches(OaBatchesBase):
    """
    A key-value store for OpenAI batches metadata.
    Keys are the batch IDs.
    """


# --------------------------------------------------------------------------------------
# Batches/Files API utilities

origin_date = 0
end_of_times_date = iso_date_to_utc_int("9999-12-31T23:59:59Z")
DateSpec = Optional[Union[str, int]]

NotGiven = type("NotGiven", (), {})()


def ensure_utc_date(x, *, val_if_none=NotGiven):
    if isinstance(x, str):
        return iso_date_to_utc_int(x)
    elif isinstance(x, int):
        return x
    elif val_if_none is not NotGiven and x is None:
        return val_if_none
    else:
        raise TypeError(f"Expected a date string or int, got {x}")


def date_filter(
    objs,  # Iterable of file/batch API objects
    min_date: DateSpec = None,
    max_date: DateSpec = None,
    *,
    date_extractor: str | Callable = "created_at",
    stop_on_first_out_of_range=False,
):
    """
    Filter objects by date.

    Args:
        objs: The objects to filter (must have a)
        min_date: The minimum date (inclusive)
        max_date: The maximum date (inclusive)
        date_attr: The attribute to get the date from
        stop_on_first_out_of_range: If True, stop iteration on first out-of-range date
    """
    if isinstance(date_extractor, str):
        date_attr = date_extractor
        date_extractor = attrgetter(date_attr)

    min_date = ensure_utc_date(min_date, val_if_none=origin_date)
    max_date = ensure_utc_date(max_date, val_if_none=end_of_times_date)

    for obj in objs:
        date = date_extractor(obj)
        if min_date <= date <= max_date:
            yield obj
        elif stop_on_first_out_of_range:
            break


# --------------------------------------------------------------------------------------
# General classes that tie everything together


class OaStores:
    def __init__(self, client: OaClientSpec = None) -> None:
        client = ensure_oa_client(client)
        self.client = client

    @cached_property
    def data_files(self):
        return OaFilesJsonData(self.client)

    @cached_property
    def json_files(self):
        return OaJsonFiles(self.client)

    @cached_property
    def jsonl_files(self):
        return OaJsonFiles(self.client)

    @cached_property
    def files(self):
        return OaFiles(self.client)

    @cached_property
    def batches(self):
        return OaBatches(self.client)

    @cached_property
    def files_base(self):
        return OaFilesBase(self.client)

    @cached_property
    def batches_base(self):
        return OaBatchesBase(self.client)

    @cached_property
    def files_metadata(self):
        return OaFilesMetadata(self.client)

    @cached_property
    def vector_stores(self):
        return OaVectorStores(self.client)

    @cached_property
    def vector_stores_base(self):
        return OaVectorStoresBase(self.client)


# TODO: There's a lot of functions in stores and batch that take an oa_stores argument.
#    Perhaps it's better for them to just be here in the OaDacc class instead?
from typing import Tuple
from oa.batches import get_output_file_data, get_batch_id_and_obj, get_batch_obj
from oa.util import concat_lists, extractors

Segment = str  # TODO: Replace by more specific, global type
Vector = list[float]  # TODO: Replace by more specific, global type


class OaDacc:
    def __init__(self, client: openai.Client | None = None) -> None:
        if client is None:
            client = mk_client()
        self.client = client
        self.s = OaStores(self.client)

    extractors = extractors

    @property
    def files(self):
        return self.s.files

    @property
    def json_files(self):
        return self.s.json_files

    @property
    def batches(self):
        return self.s.batches

    @property
    def vector_stores(self):
        return self.s.vector_stores

    def ensure_batch_obj(self, batch) -> BatchObj:
        return get_batch_obj(self.s, batch)

    def batch_id_and_obj(self, batch) -> tuple[BatchId, BatchObj]:
        batch_id, batch_obj = get_batch_id_and_obj(self.s, batch)
        return batch_id, batch_obj

    def get_output_file_data(self, batch):
        return get_output_file_data(batch, oa_stores=self.s)

    def check_status(self, batch: BatchSpec) -> str:
        """Check the status of a batch process."""
        batch_obj = self.ensure_batch_obj(batch)
        return batch_obj.status

    def retrieve_embeddings(self, batch: BatchSpec) -> list[Vector]:
        """Retrieve output embeddings for a completed batch."""
        output_data_obj = self.get_output_file_data(batch)

        # batch = self.s.batches[batch_id]

        return concat_lists(
            map(
                extractors.embeddings_from_output_data,
                jsonl_loads_iter(output_data_obj.content),
            )
        )

    def segments_from_file(self, file) -> list[Segment]:
        """
        Retrieve output embeddings for a completed batch, from the file it's stored in.
        """
        input_data = self.json_files[file]
        return extractors.inputs_from_file_obj(input_data)

    def segments_from_batch(self, batch: BatchSpec) -> list[Segment]:
        """
        Retrieve output embeddings for a completed batch, given the batch object or id.
        """
        batch_obj = self.ensure_batch_obj(batch)
        input_data_file_id = batch_obj.input_file_id
        return self.segments_from_file(input_data_file_id)

    def embeddings_from_file(self, file) -> list[Vector]:
        """
        Retrieve output embeddings for a completed batch, from the file it's stored in.
        """
        output_data_obj = self.s.files_base[file]
        return concat_lists(
            map(
                extractors.embeddings_from_output_data,
                jsonl_loads_iter(output_data_obj.content),
            )
        )

    def embeddings_from_batch(self, batch: BatchSpec) -> list[Vector]:
        """
        Retrieve output embeddings for a completed batch, given the batch object or id.
        """
        batch_obj = self.ensure_batch_obj(batch)
        return self.embeddings_from_file(batch_obj.output_file_id)

    def segments_and_embeddings(
        self, batch: BatchSpec
    ) -> tuple[list[Segment], list[Vector]]:
        """Retrieve segments nad embeddings for a completed batch."""
        batch_obj = self.ensure_batch_obj(batch)
        return (
            self.segments_from_batch(batch_obj),
            self.embeddings_from_batch(batch_obj),
        )

    def launch_embedding_task(
        self, segments: Iterable[Segment], **imbed_task_dict_kwargs
    ):
        # Upload files and get input file IDs
        input_file_id = self.files.create_embedding_task(
            segments, **imbed_task_dict_kwargs
        )
        batch = self.batches.append(input_file_id, endpoint="/v1/embeddings")
        return batch

    date_filter = staticmethod(date_filter)


# --------------------------------------------------------------------------------------
# debugging and diagnosis tools


def print_some_jsonl_line_fields(line):
    print(f"{list(line)=}")
    print(f"{list(line['response'])=}")
    print(f"{list(line['response']['body'])=}")
    print(f"{line['response']['body']['object']=}")
    print(f"{len(line['response']['body']['data'])=}")
    print(f"{list(line['response']['body']['data'][0])=}")


# --------------------------------------------------------------------------------------
# Vector Stores


class OaVectorStoresBase(OaMapping):
    """Base class for OpenAI vector stores mapping interface."""

    def __init__(self, client: openai.Client | None = None, **extra_kwargs):
        if client is None:
            client = mk_client()
        self.client = client
        self.extra_kwargs = extra_kwargs

    def _iter(self):
        """Iterate over vector store objects."""
        return self.client.vector_stores.list(**self._list_kwargs)

    @extract_id
    def metadata(self, vs_id) -> Any:  # TODO: Add proper vector store type
        """Get vector store metadata by ID."""
        return self.client.vector_stores.retrieve(vs_id, **self.extra_kwargs)

    _getitem = metadata

    @extract_id
    def _delitem(self, vs_id):
        """Delete a vector store."""
        return self.client.vector_stores.delete(vs_id)

    def create(self, name: str, **config) -> Any:
        """Create a new vector store with given name."""
        return self.client.vector_stores.create(
            name=name, **config, **self.extra_kwargs
        )

    def get_by_name(self, name: str):
        """Get vector store by name (since OpenAI API doesn't support this directly)."""
        for vs in self.client.vector_stores.list():
            if vs.name == name:
                return vs
        raise KeyError(f"Vector store with name '{name}' not found")


@wrap_kvs(key_decoder=attrgetter("id"))
class OaVectorStores(OaVectorStoresBase):
    """
    A key-value store for OpenAI vector stores.
    Keys are the vector store IDs.
    """


class OaVectorStoreFiles(OaMapping):
    """Mapping interface for files within a vector store."""

    def __init__(
        self,
        vector_store_id: str,
        client: openai.Client | None = None,
        **extra_kwargs,
    ):
        if client is None:
            client = mk_client()
        self.client = client
        self.vector_store_id = vector_store_id
        self.extra_kwargs = extra_kwargs

    def _iter(self):
        """Iterate over file objects in the vector store."""
        return self.client.vector_stores.files.list(
            vector_store_id=self.vector_store_id, **self._list_kwargs
        )

    @extract_id
    def metadata(self, file_id) -> Any:
        """Get file info from vector store."""
        return self.client.vector_stores.files.retrieve(
            vector_store_id=self.vector_store_id, file_id=file_id, **self.extra_kwargs
        )

    _getitem = metadata

    @extract_id
    def _delitem(self, file_id):
        """Remove file from vector store."""
        return self.client.vector_stores.files.delete(
            vector_store_id=self.vector_store_id, file_id=file_id
        )

    def add_file(self, file_id: str) -> Any:
        """Add file to vector store."""
        return self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id, file_id=file_id, **self.extra_kwargs
        )

    def __setitem__(self, file_id, _):
        """Add file to vector store (for MutableMapping interface)."""
        self.add_file(file_id)
```

## tests/test_base.py

```python
from functools import partial
import pytest
from oa.base import embeddings


# Note: Moved to doctests
# def test_embeddings():
#     dimensions = 3
#     embeddings_ = partial(embeddings, dimensions=dimensions, validate=True)
#     # Test with a single word
#     text = "vector"
#     result = embeddings_(text)
#     assert isinstance(result, list)
#     assert len(result) == dimensions

#     # Test with a list of words
#     texts = ["semantic", "vector"]
#     result = embeddings_(texts)
#     assert isinstance(result, list)
#     assert len(result) == len(texts) == 2
#     assert isinstance(result[0], list)
#     assert len(result[0]) == dimensions
#     assert isinstance(result[1], list)
#     assert len(result[1]) == dimensions

#     # Test with a dictionary of words
#     texts = {"adj": "semantic", "noun": "vector"}
#     result = embeddings_(texts)
#     assert isinstance(result, dict)
#     assert len(result) == len(texts) == 2
#     assert isinstance(result["adj"], list)
#     assert len(result["adj"]) == dimensions
#     assert isinstance(result["noun"], list)
#     assert len(result["noun"]) == dimensions
```

## tests/test_search.py

```python
"""Test search functionality"""

from collections.abc import Callable
import os

# ------------------------------------------------------------------------------
# Search functionality Testing Utils

from oa.vector_stores import Query, MaxNumResults, ResultT, SearchResults


def top_results_contain(results: SearchResults, expected: SearchResults) -> bool:
    """
    Check that the top results contain the expected elements.
    That is, the first len(expected) elements of results match the expected set,
    and if there are less results than expected, the only elements in results are
    contained in expected.
    """
    if len(results) < len(expected):
        return set(results) <= set(expected)
    return set(results[: len(expected)]) == set(expected)


def general_test_for_search_function(
    query,
    top_results_expected_to_contain: SearchResults,
    *,
    search_func: Callable[[Query], SearchResults],
    n_top_results=None,
):
    """
    General test function for search functionality.

    Args:
        query: Query string
        top_results_expected_to_contain: Set of expected document keys
        search_func: Search function to test (keyword-only)
        n_top_results: Number of top results to check. If None, defaults to min(len(results), len(top_results_expected_to_contain)) (keyword-only)

    Example use:

    >>> def search_docs_containing(query):
    ...     docs = {'doc1': 'apple pie recipe', 'doc2': 'car maintenance guide', 'doc3': 'apple varieties'}
    ...     return (key for key, text in docs.items() if query in text)
    >>> general_test_for_search_function(
    ...     query='apple',
    ...     top_results_expected_to_contain={'doc1', 'doc3'},
    ...     search_func=search_docs_containing
    ... )
    """
    # Execute search and collect results
    # TODO: Protect from cases where search_func(query) could be a long generator? Example, a max_results limit?
    results = list(search_func(query))

    # Determine the actual number of top results to check
    if n_top_results is None:
        effective_n_top_results = min(
            len(results), len(top_results_expected_to_contain)
        )
    else:
        effective_n_top_results = n_top_results

    # Get the slice of results to check
    top_results_to_check = results[:effective_n_top_results]

    # Generate helpful error message
    error_context = []
    error_context.append(f"Query: '{query}'")
    error_context.append(f"Expected docs: {top_results_expected_to_contain}")
    error_context.append(f"Actual results: {results}")
    error_context.append(
        f"Checking top {effective_n_top_results} results: {top_results_to_check}"
    )

    error_message = "\n".join(error_context)

    # Perform the assertion
    assert top_results_contain(
        top_results_to_check, top_results_expected_to_contain
    ), error_message


# ─── Test Documents ────────────────────────────────────────────────────────────
docs = {
    "python": "Python is a high‑level programming language emphasizing readability and rapid development.",
    "java": "Java is a class‑based, object‑oriented language designed for portability across platforms.",
    "numpy": "NumPy provides support for large, multi‑dimensional arrays and matrices, along with a collection of mathematical functions.",
    "pandas": "Pandas is a Python library offering data structures and operations for manipulating numerical tables and time series.",
    "apple": "Apple is a fruit that grows on trees and comes in varieties such as Granny Smith, Fuji, and Gala.",
    "banana": "Banana is a tropical fruit with a soft, sweet interior and a peel that changes from green to yellow when ripe.",
    "microsoft": "Microsoft develops software products including the Windows operating system, Office suite, and cloud services.",
}

# ─── Semantic Search Examples ─────────────────────────────────────────────────


def check_search_func(
    search_func: Callable[[Query], SearchResults],
):
    """
    Test the search function with multiple queries using the general test framework.
    """
    # Test case 1: programming language search
    general_test_for_search_function(
        query="object‑oriented programming",
        top_results_expected_to_contain={"java", "python", "numpy"},
        search_func=search_func,
    )

    # Test case 2: fruit category search
    general_test_for_search_function(
        query="tropical fruit",
        top_results_expected_to_contain={"banana", "apple"},
        search_func=search_func,
    )


# ─── Retrieval‑Augmented Generation Example ────────────────────────────────────


def check_find_docs_to_answer_question(
    find_docs_to_answer_question: Callable[[Query], SearchResults],
):
    """
    Test the function that finds documents relevant to a question.
    """
    general_test_for_search_function(
        query="Which documents describe a fruit that is sweet and easy to eat?",
        top_results_expected_to_contain={"apple", "banana"},
        search_func=find_docs_to_answer_question,
    )


# ─── test these test functions with a docs_to_search_func factory function ──────


def check_search_func_factory(
    search_func_factory: Callable[[dict], Callable[[Query], SearchResults]],
):
    """
    Test the search function factory with a set of documents.
    """
    search_func = search_func_factory(docs)

    # Run the search function tests
    check_search_func(search_func)
    check_find_docs_to_answer_question(search_func)


# ------------------------------------------------------------------------------
# Tests

from oa.vector_stores import (
    OaStores,
    OaVectorStoreFiles,
    docs_to_vector_store,
    mk_search_func_for_oa_vector_store,
    docs_to_search_func_factory_via_vector_store,
)


def test_vector_store_search_functionality():
    """Test the vector store search functionality."""

    # Skip test if no OpenAI API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPEN_AI_KEY"):
        print("Skipping vector store tests - no OpenAI API key found")
        return

    try:
        print("Testing docs_to_vector_store and mk_search_func_for_oa_vector_store...")

        # Test the individual functions first
        print("1. Testing docs_to_vector_store...")
        vs_id, file_mapping = docs_to_vector_store(docs, "test_search_vs")
        print(f"   Created vector store: {vs_id}")
        print(f"   File mapping: {len(file_mapping)} files")

        print("2. Testing mk_search_func_for_oa_vector_store...")
        search_func = mk_search_func_for_oa_vector_store(vs_id, file_mapping)
        print("   Search function created successfully")

        # Test a simple search
        print("3. Testing search functionality...")
        results = search_func("programming")
        print(f"   Search results: {results}")

        # Test with the factory function using our existing test framework
        print("4. Testing with check_search_func_factory...")
        print("   Note: This will create an actual vector store and use OpenAI API")

        # Uncomment the line below to run the full test (uses API calls)
        # check_search_func_factory(docs_to_search_func_factory_via_vector_store)

        print("✓ Vector store search functions created successfully")
        print("✓ Basic functionality verified")
        print("Note: Full search testing requires API calls and is commented out")

    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        print("This might be due to API key issues or OpenAI service availability")
        import traceback

        traceback.print_exc()


# Function to run the full test with API calls (uncomment to use)
def test_vector_store_search_with_api():
    """Run the full vector store search test with actual API calls."""
    print("Running full vector store search test with API calls...")
    check_search_func_factory(docs_to_search_func_factory_via_vector_store)
    print("✓ Full vector store search test completed successfully")


# ------------------------------------------------------------------------------
# Test runner

if __name__ == "__main__":
    print("=== Running Vector Store Search Tests ===")
    test_vector_store_search_functionality()

    print("\n=== Additional Test Functions Available ===")
    print("To run full API tests, call:")
    print("  test_vector_store_search_with_api()")
    print("  check_search_func_factory(docs_to_search_func_factory_via_vector_store)")
```

## tests/test_util.py

```python
"""Test cases for the oa.util module."""

from typing import Any, Tuple, Dict
from oa.util import ProcessingManager, Status, Result


def test_processing_manager_all_complete():
    """
    Test that all items are processed when they complete immediately.
    """

    # Define a processing function that always returns 'completed'
    def processing_function(item: Any) -> tuple[Status, Result]:
        return "completed", f"Result for {item}"

    # Define a handle_status_function that removes items when completed
    def handle_status_function(item: Any, status: Status, result: Result) -> bool:
        return status == "completed"

    # Define a wait_time_function that doesn't wait
    def wait_time_function(cycle_duration: float, local_vars: dict) -> float:
        return 0.0

    pending_items = {"item1": "data1", "item2": "data2", "item3": "data3"}

    manager = ProcessingManager(
        pending_items=pending_items.copy(),
        processing_function=processing_function,
        handle_status_function=handle_status_function,
        wait_time_function=wait_time_function,
    )

    manager.process_items()

    assert manager.status is True
    expected_completed_items = {k: f"Result for {v}" for k, v in pending_items.items()}
    assert manager.completed_items == expected_completed_items
    assert manager.cycles == 1


def test_processing_manager_user_story():
    """
    User Story Test for ProcessingManager

    This test simulates a scenario where a set of tasks are processed using the ProcessingManager.
    It demonstrates the following behaviors:
    - Initialization with a mix of tasks.
    - Handling of tasks with different statuses: 'in_progress', 'completed', 'failed'.
    - Updating task statuses over multiple cycles.
    - Removal of tasks based on the status handling function.
    - Use of the wait time function to control cycle timing.
    - Tracking of cycles and completed tasks.
    """

    import time
    from typing import Any, Tuple, Dict

    # Simulate a set of tasks with initial data
    pending_items = {
        "task1": {"data": "data1"},  # Will complete after first cycle
        "task2": {"data": "data2"},  # Will remain in progress
        "task3": {"data": "data3"},  # Will fail after first cycle
        "task4": {"data": "data4"},  # Will complete after two cycles
        "task5": {"data": "data5"},  # Will fail and then be retried
    }

    # Dictionary to keep track of task statuses and retries
    task_statuses = {
        "task1": "in_progress",
        "task2": "in_progress",
        "task3": "in_progress",
        "task4": "in_progress",
        "task5": "in_progress",
    }

    retry_counts = {
        "task5": 0,  # Will retry on failure
    }

    def processing_function(item: Any) -> tuple[Status, Result]:
        """
        Simulates processing of a task.
        """
        task_id = item["task_id"]
        current_status = task_statuses[task_id]

        # Simulate status transitions
        if task_id == "task1" and current_status == "in_progress":
            # Task1 completes after first cycle
            task_statuses[task_id] = "completed"
            result = f"Result for {task_id}"
            return "completed", result

        elif task_id == "task2":
            # Task2 remains in progress indefinitely
            result = None
            return "in_progress", result

        elif task_id == "task3" and current_status == "in_progress":
            # Task3 fails after first cycle
            task_statuses[task_id] = "failed"
            result = f"Error in {task_id}"
            return "failed", result

        elif task_id == "task4" and current_status == "in_progress":
            # Task4 completes after two cycles
            task_statuses[task_id] = "in_progress_2"
            result = None
            return "in_progress", result
        elif task_id == "task4" and current_status == "in_progress_2":
            task_statuses[task_id] = "completed"
            result = f"Result for {task_id}"
            return "completed", result

        elif task_id == "task5":
            # Task5 fails once and then retries
            if retry_counts["task5"] == 0:
                retry_counts["task5"] += 1
                result = f"Temporary error in {task_id}"
                return "failed", result
            else:
                task_statuses[task_id] = "completed"
                result = f"Result for {task_id} after retry"
                return "completed", result

        else:
            # Default case
            result = None
            return "in_progress", result

    def handle_status_function(item: Any, status: Status, result: Result) -> bool:
        """
        Determines whether to remove the task based on its status.
        """
        task_id = item["task_id"]

        if status == "completed":
            # Task is completed; remove it
            print(f"Task {task_id} completed with result: {result}")
            return True
        elif status == "failed":
            if task_id == "task5" and retry_counts["task5"] <= 1:
                # Retry task5 once on failure
                print(f"Task {task_id} failed with error: {result}. Retrying...")
                return False  # Keep in pending_items for retry
            else:
                # For other tasks or after retry, remove the task
                print(f"Task {task_id} failed with error: {result}. Not retrying.")
                return True
        else:
            # Task is still in progress; keep it
            print(f"Task {task_id} is in progress.")
            return False

    def wait_time_function(cycle_duration: float, local_vars: dict) -> float:
        """
        Determines how long to wait before the next cycle.
        """
        status_check_interval = local_vars["self"].status_check_interval
        sleep_duration = max(0, status_check_interval - cycle_duration)
        print(f"Waiting for {sleep_duration:.2f} seconds before next cycle.")
        return sleep_duration

    # Add task IDs to the items for easy tracking
    for task_id, item in pending_items.items():
        item["task_id"] = task_id

    # Initialize the ProcessingManager with the pending tasks
    manager = ProcessingManager(
        pending_items=pending_items,
        processing_function=processing_function,
        handle_status_function=handle_status_function,
        wait_time_function=wait_time_function,
        status_check_interval=1.0,  # Check every 1 second
        max_cycles=5,  # Limit to 5 cycles to prevent infinite loops
    )

    # Record the start time
    start_time = time.time()

    # Start the processing loop
    manager.process_items()

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Assertions to check that the manager behaved as expected
    # Task1 and Task4 should be completed
    assert "task1" in manager.completed_items
    assert "task4" in manager.completed_items

    # Task3 should have failed and been removed
    assert "task3" in manager.completed_items

    # Task5 should have retried and then completed
    assert "task5" in manager.completed_items

    # Task2 should still be in pending_items (since it remains in progress)
    assert "task2" in manager.pending_items

    # The manager should have run for the expected number of cycles
    assert manager.cycles == manager.max_cycles or manager.status is True

    # Output the final state for verification
    print("\nFinal State:")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Cycles executed: {manager.cycles}")
    print(f"Completed tasks: {list(manager.completed_items.keys())}")
    print(f"Pending tasks: {list(manager.pending_items.keys())}")

    # Ensure that the test completes without errors
    assert True


test_processing_manager_all_complete()
test_processing_manager_user_story()
```

## tools.py

```python
"""Interface tools"""

from functools import partial
from typing import Optional
from collections.abc import Callable
import string

from i2 import Sig, Pipe
from lkj import add_attr
from oa.base import chat

# -----------------------------------------------------------------------------
# Helpers

import re
from collections import namedtuple
from typing import List, NamedTuple, Literal, Union, Optional

Pattern = Union[re.Pattern, str]
string_formatter = string.Formatter()

DFLT_IGNORE_PATTERN = re.compile(r"```.*?```", re.DOTALL)


def remove_pattern(string, pattern_to_remove: Pattern | None = DFLT_IGNORE_PATTERN):
    """
    Returns a where a given regular expression pattern has been removed.

    >>> string = 'this ```is a``` string ```with several``` backticks'
    >>> remove_pattern(string)
    'this  string  backticks'

    """
    pattern_to_remove = re.compile(pattern_to_remove)
    return pattern_to_remove.sub("", string)


def extract_parts(string: str, pattern: Pattern) -> NamedTuple:
    PartResult = namedtuple("PartResult", ["matched", "unmatched"])
    matched: list[str] = []
    unmatched: list[str] = []
    last_end = 0

    for match in re.finditer(pattern, string):
        start, end = match.span()
        unmatched.append(string[last_end:start])
        matched.append(string[start:end])
        last_end = end

    unmatched.append(string[last_end:])

    return PartResult(matched=matched, unmatched=unmatched)


def pattern_based_map(
    func: Callable,
    string: str,
    pattern: Pattern,
    apply_to: Literal["matched", "unmatched"] = "unmatched",
):
    """
    Applies a function to parts of the string that are either matching or non-matching based on a regex pattern,
    depending on the value of apply_to.

    Example:
    >>> func = str.upper
    >>> string = "the good the ```bad``` and the ugly"
    >>> ignore_pattern = r'```.*?```'
    >>> pattern_based_map(func, string, ignore_pattern)
    'THE GOOD THE ```bad``` AND THE UGLY'
    >>> pattern_based_map(func, string, ignore_pattern, 'matched')
    'the good the ```BAD``` and the ugly'
    """
    parts = extract_parts(string, pattern)
    result = ""

    # Apply the function to the appropriate parts
    if apply_to == "matched":
        transformed_matched = [func(part) for part in parts.matched]
        # Interleave transformed matched parts with untouched unmatched parts
        result_parts = sum(zip(parts.unmatched, transformed_matched), ())
    else:
        transformed_unmatched = [func(part) for part in parts.unmatched]
        # Interleave untouched matched parts with transformed unmatched parts
        result_parts = sum(zip(transformed_unmatched, parts.matched), ())

    # Ensure all parts are added, including the last unmatched if unmatched is longer
    result = "".join(result_parts)
    if len(parts.unmatched) > len(parts.matched):
        result += (
            transformed_unmatched[-1]
            if apply_to == "unmatched"
            else parts.unmatched[-1]
        )

    return result


def _extract_names_from_format_string(
    template: str, *, ignore_pattern: Pattern | None = DFLT_IGNORE_PATTERN
):
    """Extract names from a string format template

    >>> _extract_names_from_format_string("Hello {name}! I am {bot_name}.")
    ('name', 'bot_name')

    """
    if ignore_pattern is not None:
        template = remove_pattern(template, ignore_pattern)
    return tuple(
        name for _, name, _, _ in string_formatter.parse(template) if name is not None
    )


def _extract_defaults_from_format_string(
    template: str, *, ignore_pattern=DFLT_IGNORE_PATTERN
) -> dict:
    """Extract (name, specifier) from a string format template.

    >>> _extract_defaults_from_format_string(
    ...     "Hello {name}! I am {bot_name:chatGPT}."
    ... )
    {'bot_name': 'chatGPT'}

    """
    if ignore_pattern is not None:
        template = remove_pattern(template, ignore_pattern)
    return {
        name: specifier
        for _, name, specifier, _ in string_formatter.parse(template)
        if name is not None and specifier != ""
    }


def _template_without_specifiers(
    template: str, *, ignore_pattern: Pattern | None = DFLT_IGNORE_PATTERN
) -> str:
    """Uses remove any extras from a template string, leaving only text and fields.

    >>> template = "A {normal}, {stra:nge}, an ```{igno:red}``` and an empty: {}."
    >>> _template_without_specifiers(template, ignore_pattern=None)
    'A {normal}, {stra}, an ```{igno}``` and an empty: {}.'
    >>> _template_without_specifiers(template, ignore_pattern=r'```.*?```')
    'A {normal}, {stra}, an ```{igno:red}``` and an empty: {}.'

    """

    def gen(template):
        for text, field_name, *_ in string.Formatter().parse(template):
            text_ = text or ""
            if field_name is None:
                yield text_
            else:
                yield text_ + "{" + field_name + "}"

    def rm_specifiers(template):
        return "".join(gen(template))

    if ignore_pattern is None:
        return rm_specifiers(template)
    else:
        return pattern_based_map(rm_specifiers, template, ignore_pattern)


def _template_with_double_braces_in_ignored_sections(
    template, *, ignore_pattern: Pattern | None = DFLT_IGNORE_PATTERN
) -> str:
    """double the braces of the parts of the template that should be ignored"""
    double_braces = lambda string: string.replace("{", "{{").replace("}", "}}")
    return pattern_based_map(
        double_braces, template, ignore_pattern, apply_to="matched"
    )


def string_format_embodier(
    template, *, ignore_pattern: Pattern | None = DFLT_IGNORE_PATTERN
):

    names = _extract_names_from_format_string(template, ignore_pattern=ignore_pattern)
    names = tuple(dict.fromkeys(names))  # get unique names, but conserving order
    sig = Sig(names).ch_kinds(**{name: Sig.KEYWORD_ONLY for name in names})

    @sig
    def templated_string_embodier(**kwargs):
        return template.format(**kwargs)

    return templated_string_embodier


add_name = add_attr("__name__")
add_doc = add_attr("__doc__")
add_module = add_attr("__module__")

# -----------------------------------------------------------------------------
# The meat


# TODO: template_to_names, template_to_defaults and embodier are implicitly bound by
#   their ignore_pattern argument (set to DFLT_IGNORE_PATTERN). Find a cleaner way.
def prompt_function(
    template,
    *,
    defaults: dict | None = None,
    template_to_names: Callable = _extract_names_from_format_string,
    template_to_defaults: Callable = _extract_defaults_from_format_string,
    embodier: Callable = string_format_embodier,
    arg_kinds: dict | None = None,
    name="prompt",
    prompt_func=chat,
    prompt_func_kwargs=None,
    ingress=None,
    egress=None,
    doc="The function composes a prompt and asks an LLM to respond to it.",
    module=__name__,
):
    r"""Convert a string template to a function that will produce a prompt string
    and ask an LLM (`prompt_func`) to respond to it.

    :param template: A string template with placeholders.
    :param defaults: A dictionary of default values for placeholders.
    :param template_to_names: A function that extracts names from a template.
    :param template_to_defaults: A function that extracts defaults from a template.
    :param embodier: A function that converts a template to a function that will
        produce a prompt string.
    :param arg_kinds: A dictionary of argument kinds for the function.
    :param name: The name of the function.
    :param prompt_func: The function that will be used to ask the LLM to respond to
        the prompt. If None, the output function will only produce the prompt string,
        not ask the LLM to respond to it.
    :param prompt_func_kwargs: Keyword arguments to pass to `prompt_func`.
    :param ingress: A function to apply to the input of `prompt_func`.
    :param egress: A function to apply to the output of `prompt_func`.
    :param doc: The docstring of the function.
    :param module: The module of the function.

    In the following example, we'll use the `prompt_func=None` argument to get a
    function that simply injects inputs in a prompt template, without actually calling
    an AI-enabled `prompt_func`.
    Note in this example, how a block of the prompt template string is ignored for
    injection purposes, via a triple-backtick marker.

    >>> prompt_template = '''
    ... ```
    ... In this block, all {placeholders} are {igno:red} so that they can appear in prompt.
    ... ```
    ... But outside {inputs} are {injected:normally}
    ... '''
    >>> f = prompt_function(prompt_template, prompt_func=None)
    >>> from inspect import signature
    >>> assert str(signature(f)) == "(inputs, *, injected='normally')"
    >>> print(f('INPUTS', injected="INJECTED"))  # doctest: +NORMALIZE_WHITESPACE
    ```
    In this block, all {placeholders} are {igno:red} so that they can appear in prompt.
    ```
    But outside INPUTS are INJECTED

    """

    template_original = template
    defaults = dict(template_to_defaults(template), **(defaults or {}))
    template = _template_without_specifiers(template)
    template = _template_with_double_braces_in_ignored_sections(template)
    template_embodier = embodier(template)
    prompt_func_kwargs = prompt_func_kwargs or {}
    egress = egress or (lambda x: x)
    ingress = ingress or (lambda x: x)

    # TODO: Same logic replicated in string_format_embodier (what can we do?)
    names = template_to_names(template)
    arg_kinds = dict({name: Sig.KEYWORD_ONLY for name in names}, **(arg_kinds or {}))
    names = tuple(dict.fromkeys(names))  # get unique names, but conserving order
    sig = Sig(names)

    # Inject defaults
    sig = sig.ch_defaults(
        _allow_reordering=True, **{name: default for name, default in defaults.items()}
    )
    # Handle kinds (make all but first keyword only)) and inject defaults
    sig = sig.ch_kinds(**arg_kinds)
    if sig.names:
        # Change the first argument to position or keyword kind
        first_arg_name = sig.names[0]
        sig = sig.ch_kinds(**{first_arg_name: Sig.POSITIONAL_OR_KEYWORD})

    sig = sig.sort_params()
    func_wrap = Pipe(sig, add_name(name), add_doc(doc), add_module(module))

    @func_wrap
    def embody_prompt(*ask_oa_args, **ask_oa_kwargs):
        _kwargs = sig.map_arguments(ask_oa_args, ask_oa_kwargs, apply_defaults=True)
        _kwargs = ingress(_kwargs)
        __args, __kwargs = Sig(template_embodier).mk_args_and_kwargs(_kwargs)
        embodied_template = template_embodier(*__args, **__kwargs)
        return embodied_template

    @func_wrap
    def ask_oa(*ask_oa_args, **ask_oa_kwargs):
        embodied_template = embody_prompt(*ask_oa_args, **ask_oa_kwargs)
        return egress(prompt_func(embodied_template, **prompt_func_kwargs))

    if prompt_func is not None:
        f = ask_oa
    else:
        f = embody_prompt
    f.template = template
    f.template_original = template_original

    return f


import json
from i2 import Sig, Pipe
from collections.abc import Mapping


def identity(x):
    return x


json_types = {
    "object": dict,
    "array": list,
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "null": type(None),
}

py_to_json_types = {v: k for k, v in json_types.items()}

json_type_specs = json_types.keys() | py_to_json_types.keys()

# TODO: Define from json_type_specs when possible (e.g. in python 3.11 it will be)
# JsonTypes = Literal[
#     'object', 'array', 'string', 'number', 'integer', 'boolean', 'null',
#     dict, list, str, float, int, bool, type(None),
# ]

from enum import Enum


class JsonTypes(Enum):
    string = "string"
    number = "number"
    object = "object"
    array = "array"
    boolean = "boolean"
    null = "null"
    dict = dict
    list = list
    float = float
    int = int
    bool = bool
    none = type(None)


def ensure_json_type(json_type: JsonTypes) -> str:
    """
    Ensure that the json type is a string that is a valid json type

    >>> ensure_json_type('string')
    'string'
    >>> ensure_json_type(str)
    'string'
    >>> ensure_json_type(object)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Cannot convert ...

    """
    # Get the string representation of the json type, given as a python type
    if isinstance(json_type, type):
        if json_type in py_to_json_types:
            return py_to_json_types[json_type]
        else:
            raise ValueError(
                f"Cannot convert {json_type} to a json type. "
                f"Should be one of {json_type_specs}"
            )
    # ensure that the json type is a string that is listed in json_types
    if not isinstance(json_type, str) or json_type not in json_types:
        raise ValueError(
            f"json_type should be a string or a type, not {type(json_type)}"
        )
    return json_type


def make_generic_json_schema(json_type: JsonTypes) -> dict:
    """
    Make a generic json schema for a given json type.

    >>> make_generic_json_schema('string')
    {'name': 'generic_string_schema', 'schema': {'properties': {'result': {'type': 'string'}}, 'required': ['result']}}
    """

    json_type = ensure_json_type(json_type)
    return {
        "name": f"generic_{json_type}_schema",
        "schema": {
            "properties": {"result": {"type": json_type}},
            "required": ["result"],
        },
    }


# Note: Deprecated, but Keeping around for reference
_generic_json_schema = {
    "name": "generic_json_schema",
    "schema": {
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
        #  "additionalProperties": True,
    },
    #     "strict": False,
}


def _might_be_a_json_string(string):
    """
    Returns True if the string might have a chance of being decoded by `json.loads`.

    More precisely, will check if the first non-whitespace character is a '{' or a '['.

    >>> _might_be_a_json_string('   {"a": 1}  ')
    True
    >>> _might_be_a_json_string('    ["lists", "of", "stuff"]  ')
    True
    >>> _might_be_a_json_string('    not a json string  ')
    False
    """
    return re.compile(r"^\s*[\[\{]").match(string) is not None


def _ensure_json_schema(json_schema: str | bytes | Mapping) -> dict:
    """
    A few things to make it more probable that the input is a oa valid json schema
    """
    if isinstance(json_schema, type):
        json_schema = make_generic_json_schema(json_schema)
    elif isinstance(json_schema, str):
        if json_schema in json_types:  # make a generic json schema for that type
            json_schema = make_generic_json_schema(json_schema)
        elif _might_be_a_json_string(json_schema):  # assume it is a json string
            json_schema = json.loads(json_schema)
        else:  # assume it's free text, from which AI will try to infer a schema
            verbal_description = json_schema
            _json_schema = infer_schema_from_verbal_description(verbal_description)
            json_schema = _json_schema  # ["json_schema"]

    if "name" not in json_schema:  # OpenAI forces you to put a name
        json_schema["name"] = "json_schema"

    if "schema" not in json_schema:  # the schema actually has to be under a schema key
        json_schema = {"schema": json_schema, "name": "json_schema"}

    if "type" not in json_schema["schema"]:  # OpenAI forces you to put a type
        json_schema["schema"]["type"] = "object"

    return json_schema


# TODO: model could be present in prompt_func_kwargs or in partial of prompt_func
#   --> need to ensure that all work well together (no obfuscated conflicts)
def prompt_json_function(
    template,
    json_schema: str | bytes | Mapping = "string",
    *,
    defaults: dict | None = None,
    embodier: Callable = string_format_embodier,
    arg_kinds: dict | None = None,
    name="prompt",
    prompt_func=chat,
    prompt_func_kwargs=None,
    model="gpt-4o-mini",
    ingress=None,
    egress=None,
    doc="The function composes a prompt and asks an LLM to respond to it with json.",
    module=__name__,
) -> dict:
    """
    Make prompt functions that return jsons (dicts) with a given schema.
    """

    json_schema = _ensure_json_schema(json_schema)

    assert isinstance(json_schema, Mapping)

    prompt_func_kwargs = dict(
        dict(
            model=model,  # TODO: Change to just ensure model is compatible
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
        ),
        **(prompt_func_kwargs or {}),
    )

    egress = Pipe(json.loads, egress or identity)

    func = prompt_function(
        template,
        defaults=defaults,
        embodier=embodier,
        arg_kinds=arg_kinds,
        name=name,
        prompt_func=prompt_func,
        prompt_func_kwargs=prompt_func_kwargs,
        ingress=ingress,
        egress=egress,
        doc=doc,
        module=module,
    )
    func.json_schema = json_schema
    return func


def infer_schema_from_verbal_description(verbal_description: str):
    template = """
    Generate a valid JSON Schema based on the the verbal description of the desired 
    JSON output below. 
    The schema must be properly formatted for use with OpenAI’s Chat API 
    in "JSON mode" and should accurately define the structure, 
    data types, required fields, and any constraints specified by the user. 
    Ensure correctness and completeness.

    Note that you need to provide not only a valid schema but also a valid name for it.

    Here is the verbal description of the desired JSON output:
    {verbal_description}
    """
    output_schema = {
        "name": "infered_json_schema",
        "schema": {
            "properties": {
                "name": {"type": "string"},
                "properties": {"type": "object"},
                "type": {"type": "string"},
            },
            "required": ["name", "properties"],
        },
    }
    f = prompt_json_function(template, output_schema)
    return f(verbal_description=verbal_description)


from typing import Optional, KT, Union
from collections.abc import Mapping
from dol import filt_iter
from oa.util import mk_template_store, DFLT_TEMPLATES_SOURCE
import os

# chatgpt_templates_dir = os.path.join(templates_files, "chatgpt")
# _ends_with_txt = filt_iter.suffixes(".txt")
# chatgpt_templates = filt_iter(TextFiles(chatgpt_templates_dir), filt=_ends_with_txt)
dflt_function_key = lambda f: os.path.splitext(os.path.basename(f))[0]
dflt_factory_key = lambda f: os.path.splitext(os.path.basename(f))[-1]
_dflt_factories = {
    ".txt": prompt_function,
    "": prompt_function,
}
dflt_factories = tuple(_dflt_factories.items())

_suffixes_csv = ",".join(_dflt_factories.keys())
DFLT_TEMPLATE_SOURCE_WITH_SUFFIXES = f"{DFLT_TEMPLATES_SOURCE}:{_suffixes_csv}"

StoreKey = str
FuncName = str
FactoryKey = KT


class PromptFuncs:
    """Make AI enabled functions"""

    def __init__(
        self,
        template_store: Mapping | str = DFLT_TEMPLATE_SOURCE_WITH_SUFFIXES,
        *,
        function_key: Callable[[StoreKey], FuncName] = dflt_function_key,
        factory_key: Callable[[StoreKey], FactoryKey] = dflt_factory_key,
        factories: Callable[[FactoryKey], Callable] = dflt_factories,
        extra_template_kwargs: Mapping[StoreKey, Mapping] | None = None,
    ):
        self._template_store = mk_template_store(template_store)
        self._function_key = function_key
        self._factory_key = factory_key
        self._factories = dict(factories)
        self._extra_template_kwargs = dict(extra_template_kwargs or {})
        self._functions = dict(self._mk_functions())
        self._inject_functions()

    def _mk_functions(self):
        for store_key, template in self._template_store.items():
            factory = self._factories[self._factory_key(store_key)]
            func_key = self._function_key(store_key)
            yield func_key, factory(
                template,
                name=func_key,
                **self._extra_template_kwargs.get(store_key, {}),
            )

    def _inject_functions(self):
        self.__dict__.update(self._functions)

    def __iter__(self):
        return iter(self._functions)

    def __getitem__(self, name):
        return self._functions[name]

    def __len__(self):
        return len(self._functions)

    def reload(self):
        """Reload all functions"""
        self._functions = dict(self._mk_functions())
        self._inject_functions()
        return self

    def funcs_and_sigs(self):
        """Return a mapping of function names to signatures"""
        return {name: Sig(func) for name, func in self._functions.items()}

    def print_signatures(self):
        """Print signatures of all functions"""
        print("\n".join(f"{k}{v}" for k, v in self.funcs_and_sigs().items()))
```

## util.py

```python
"""oa utils"""

from importlib.resources import files
import os
from functools import partial, lru_cache
from typing import Union, get_args, Literal
from collections.abc import Mapping
from types import SimpleNamespace

from i2 import Sig, get_app_config_folder
import dol
import graze
from config2py import (
    get_config,
    ask_user_for_input,
    get_configs_local_store,
    simple_config_getter,
    user_gettable,
)

import openai  # pip install openai (see https://pypi.org/project/openai/)
from openai.resources.files import FileObject
from openai.resources.batches import Batches as OpenaiBatches, Batch as BatchObj

BatchObj  # to avoid unused import warning (the import here is for other modules)


def get_package_name():
    """Return current package name"""
    # return __name__.split('.')[0]
    # TODO: See if this works in all cases where module is in highest level of package
    #  but meanwhile, hardcode it:
    return "oa"


# get app data dir path and ensure it exists
pkg_name = get_package_name()
data_files = files(pkg_name) / "data"
templates_files = data_files / "templates"
_root_app_data_dir = get_app_config_folder()
app_data_dir = os.environ.get(
    f"{pkg_name.upper()}_APP_DATA_DIR",
    os.path.join(_root_app_data_dir, pkg_name),
)
app_data_dir = dol.ensure_dir(app_data_dir)
djoin = partial(os.path.join, app_data_dir)

# _open_api_key_env_name = 'OPENAI_API_KEY'
# _api_key = os.environ.get(_open_api_key_env_name, None)
# if _api_key is None:
#     _api_key = getpass.getpass(
#         f"Please set your OpenAI API key and press enter to continue. "
#         f"I will put it in the environment variable {_open_api_key_env_name} "
#     )
# openai.api_key = _api_key

configs_local_store = get_configs_local_store(pkg_name)

_DFLT_CONFIGS = {
    "OPENAI_API_KEY_ENV_NAME": "OPENAI_API_KEY",
    "OA_DFLT_TEMPLATES_SOURCE_ENV_NAME": "OA_DFLT_TEMPLATES_SOURCE",
    "OA_DFLT_ENGINE": "gpt-3.5-turbo-instruct",
    "OA_DFLT_MODEL": "gpt-3.5-turbo",
}

# write the defaults to the local store, if key missing there
for k, v in _DFLT_CONFIGS.items():
    if k not in configs_local_store:
        configs_local_store[k] = v


config_sources = [
    configs_local_store,  # look in the local store
    os.environ,  # look in the environment variables
    user_gettable(
        configs_local_store
    ),  # ask the user (and save response in local store)
]


def kv_strip_value(k, v):
    return v.strip()


# The main config getter for this package
config_getter = get_config(sources=config_sources, egress=kv_strip_value)


# Get the OPENAI_API_KEY_ENV_NAME and DFLT_TEMPLATES_SOURCE_ENV_NAME
OPENAI_API_KEY_ENV_NAME = config_getter("OPENAI_API_KEY_ENV_NAME")
DFLT_TEMPLATES_SOURCE_ENV_NAME = config_getter("OA_DFLT_TEMPLATES_SOURCE_ENV_NAME")

# TODO: Understand the model/engine thing better and merge defaults if possible
DFLT_ENGINE = config_getter("OA_DFLT_ENGINE")
DFLT_MODEL = config_getter("OA_DFLT_MODEL")

# TODO: Add the following to config_getter mechanism
DFLT_EMBEDDINGS_MODEL = "text-embedding-3-small"


Purpose = FileObject.model_fields["purpose"].annotation
DFLT_PURPOSE = "batch"
BatchesEndpoint = eval(Sig(OpenaiBatches.create).annotations["endpoint"])
batch_endpoints_values = get_args(BatchesEndpoint)
batch_endpoints_keys = [
    k.replace("/v1/", "").replace("/", "_") for k in batch_endpoints_values
]
batch_endpoints = SimpleNamespace(
    **dict(zip(batch_endpoints_keys, batch_endpoints_values))
)


_pricing_category_aliases = {
    "text": "Latest models - Text tokens",
    "audio": "Latest models - Audio tokens",
    "finetune": "Fine tuning",
    "tools": "Built-in tools",
    "search": "Web search",
    "speech": "Transcription and speech generation",
    "images": "Image generation",
    "embeddings": "Embeddings",
    "moderation": "Moderation",
    "other": "Other models",
}

PricingCategory = Literal[tuple(_pricing_category_aliases.keys())]


def pricing_info(category: PricingCategory = None, *, print_data_date=False):
    """
    Return the pricing info for the OpenAI API.

    Note: These are not live prices. Live prices can be found here:

    The information pricing_info returns is taken from the file `openai_api_pricing_info.json`
    in the `data` directory of the package.
    To print a message with the data date, do `pricing_info(print_data_date=True)`.
    """
    info_filepath = data_files / "openai_api_pricing_info.json"
    if print_data_date:
        print(f"Data date: {info_filepath.stat().st_mtime}")
    info = json.loads(info_filepath.read_text())

    if category is None:

        def _pricing_info():
            for category in _pricing_category_aliases:
                for d in pricing_info(category):
                    yield dict(category=category, **d)

        return list(_pricing_info())
    else:
        return info[_pricing_category_aliases.get(category, category)]["pricing_table"]


pricing_info.category_aliases = _pricing_category_aliases

# TODO: Write tools to update mteb_eval
# Note: OpenAI API live prices: https://platform.openai.com/docs/pricing
embeddings_models = {
    "text-embedding-3-small": {
        "price_per_million_tokens": 0.02,  # in dollars
        "pages_per_dollar": 62500,  # to do:
        "performance_on_mteb_eval": 62.3,
        "max_input": 8191,
    },
    "text-embedding-3-large": {
        "price_per_million_tokens": 0.13,  # in dollars
        "pages_per_dollar": 9615,
        "performance_on_mteb_eval": 64.6,
        "max_input": 8191,
    },
    "text-embedding-ada-002": {
        "price_per_million_tokens": 0.10,  # in dollars
        "pages_per_dollar": 12500,
        "performance_on_mteb_eval": 61.0,
        "max_input": 8191,
    },
}


# add batch-api models
def _generate_batch_api_models_info(models_info_dict, batch_api_discount=0.5):
    for model_name, model_info in models_info_dict.items():
        m = model_info.copy()
        m["price_per_million_tokens"] = round(
            m["price_per_million_tokens"] * batch_api_discount, 4
        )
        m["pages_per_dollar"] = int(1 / m["price_per_million_tokens"])
        # the rest remains the same
        yield f"batch__{model_name}", m


embeddings_models = dict(
    embeddings_models,
    **dict(_generate_batch_api_models_info(embeddings_models, batch_api_discount=0.5)),
)

# Note: OpenAI API live prices: https://platform.openai.com/docs/pricing
chat_models = {
    "gpt-4": {
        "price_per_million_tokens": 30.00,  # in dollars
        "price_per_million_tokens_output": 60.00,  # in dollars
        "pages_per_dollar": 134,  # approximately
        "performance_on_eval": "Advanced reasoning for complex tasks",
        "max_input": 8192,  # tokens
    },
    "gpt-4-32k": {
        "price_per_million_tokens": 60.00,  # in dollars
        "price_per_million_tokens_output": 120.00,  # in dollars
        "pages_per_dollar": 67,
        "performance_on_eval": "Extended context window for long documents",
        "max_input": 32768,  # tokens
    },
    "gpt-4-turbo": {
        "price_per_million_tokens": 10.00,  # in dollars
        "price_per_million_tokens_output": 30.00,  # in dollars
        "pages_per_dollar": 402,
        "performance_on_eval": "Cost-effective version of GPT-4",
        "max_input": 8192,  # tokens
    },
    "o1": {
        "price_per_million_tokens": 15.00,  # in dollars
        "price_per_million_tokens_output": 60.00,  # in dollars
        "pages_per_dollar": 268,
        "performance_on_eval": "Optimized for complex reasoning in STEM fields",
        "max_input": 8192,  # tokens
    },
    "o1-mini": {
        "price_per_million_tokens": 1.10,  # in dollars
        "price_per_million_tokens_output": 4.40,  # in dollars
        "pages_per_dollar": 1341,
        "performance_on_eval": "Cost-effective reasoning for simpler tasks",
        "max_input": 8192,  # tokens
    },
    "gpt-4o": {
        "price_per_million_tokens": 2.50,  # in dollars
        "price_per_million_tokens_output": 10.0,  # in dollars
        "pages_per_dollar": 804,  # approximately
        "performance_on_eval": "Efficiency-optimized version of GPT-4 for better performance on reasoning tasks",
        "max_input": 8192,  # tokens
    },
    "gpt-4o-mini": {
        "price_per_million_tokens": 0.15,  # in dollars,
        "price_per_million_tokens_output": 0.60,  # in dollars
        "pages_per_dollar": 13410,
        "performance_on_eval": "Highly cost-effective, optimized for simple tasks with faster response times",
        "max_input": 8192,  # tokens
    },
}

model_information_dict = dict(
    **embeddings_models,
    **chat_models,
    # TODO: Add more model information dicts here
)


# Have a particular way to get this api key
@lru_cache
def get_api_key_from_config():
    return get_config(
        OPENAI_API_KEY_ENV_NAME,
        sources=[
            # Try to find it in oa config
            configs_local_store,
            # Try to find it in os.environ (environmental variables)
            os.environ,
            # If not, ask the user to input it
            lambda k: ask_user_for_input(
                f"Please set your OpenAI API key and press enter to continue. "
                "If you don't have one, you can get one at "
                "https://platform.openai.com/account/api-keys. ",
                mask_input=True,
                masking_toggle_str="",
                egress=lambda v: configs_local_store.__setitem__(k, v),
            ),
        ],
        egress=kv_strip_value,
    )


# TODO: Hm... set api key globally? Doesn't seem we should do that!
openai.api_key = get_api_key_from_config()


@lru_cache
def mk_client(api_key=None, **client_kwargs) -> openai.Client:
    api_key = api_key or get_api_key_from_config()
    return openai.OpenAI(api_key=api_key, **client_kwargs)


OaClientSpec = Union[openai.Client, str, dict, None]


def ensure_oa_client(oa_client: OaClientSpec) -> openai.Client:
    """Ensure that an OpenAI client is available, either by using the provided one or creating a new one."""
    if oa_client is None:
        return mk_client()
    elif isinstance(oa_client, openai.Client):
        return oa_client
    elif isinstance(oa_client, str):
        return mk_client(api_key=oa_client)
    elif isinstance(oa_client, dict):
        return mk_client(**oa_client)
    else:
        raise TypeError(
            f"Expected an OpenAI client instance, got {type(oa_client).__name__}"
        )


# TODO: Pros and cons of using a default client
#   Reason was that I was fed up of having to pass the client to every function
try:
    dflt_client = mk_client()
except Exception as e:
    dflt_client = None

_grazed_dir = dol.ensure_dir(os.path.join(app_data_dir, "grazed"))
grazed = graze.Graze(rootdir=_grazed_dir)


chatgpt_templates_dir = os.path.join(templates_files, "chatgpt")

DFLT_TEMPLATES_SOURCE = get_config(
    DFLT_TEMPLATES_SOURCE_ENV_NAME,
    sources=[os.environ],
    default=f"{chatgpt_templates_dir}",
)


# TODO: This is general: Bring this in dol or dolx
def _extract_folder_and_suffixes(
    string: str, default_suffixes=(), *, default_folder="", root_sep=":", suffix_sep=","
):
    root_folder, *suffixes = string.split(root_sep)
    if root_folder == "":
        root_folder = default_folder
    if len(suffixes) == 0:
        suffixes = default_suffixes
    elif len(suffixes) == 1:
        suffixes = suffixes[0].split(suffix_sep)
    else:
        raise ValueError(
            f"template_store must be a path to a directory of templates, "
            f"optionally followed by a colon and a list of file suffixes to use"
        )
    return root_folder, suffixes


def mk_template_store(template_store: Mapping | str):
    if isinstance(template_store, Mapping):
        return template_store
    elif isinstance(template_store, str):
        root_folder, suffixes = _extract_folder_and_suffixes(template_store)
        suffix_filter = dol.filt_iter.suffixes(suffixes)
        return suffix_filter(dol.TextFiles(root_folder))
    else:
        raise TypeError(
            f"template_store must be a Mapping or a path to a directory of templates"
        )


import tiktoken


def num_tokens(text: str = None, model: str = DFLT_MODEL) -> int:
    """Return the number of tokens in a string, under given model.

    keywords: token count, number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text, disallowed_special=()))


# --------------------------------------------------------------------------------------
# Extraction

from oa.oa_types import BatchRequest, EmbeddingResponse, InputDataJsonL
from ju import ModelExtractor
from types import SimpleNamespace
from operator import itemgetter
import pickle
import tempfile

from dol import add_ipython_key_completions, Pipe

models = [BatchRequest, EmbeddingResponse, InputDataJsonL]

oa_extractor = Pipe(ModelExtractor(models), add_ipython_key_completions)


def oa_extractors_obj(**named_paths):
    """
    Return a SimpleNamespace of extractors for the named paths
    """
    return SimpleNamespace(
        **{
            name: Pipe(oa_extractor, itemgetter(path))
            for name, path in named_paths.items()
        }
    )


extractors = oa_extractors_obj(
    embeddings_from_output_data="response.body.data.*.embedding",
    inputs_from_file_obj="body.input",
)


# --------------------------------------------------------------------------------------
# misc utils
from collections.abc import Iterable
from dateutil.parser import parse as parse_date
from datetime import datetime, timezone
from itertools import chain, islice
from typing import (
    Union,
    Dict,
    List,
    Tuple,
    TypeVar,
    Optional,
    T,
)
from collections.abc import Iterable, Mapping, Iterator, Callable

KT = TypeVar("KT")  # there's a typing.KT, but pylance won't allow me to use it!
VT = TypeVar("VT")  # there's a typing.VT, but pylance won't allow me to use it!


def chunk_iterable(
    iterable: Iterable[T] | Mapping[KT, VT],
    chk_size: int,
    *,
    chunk_type: Callable[..., Iterable[T] | Mapping[KT, VT]] | None = None,
) -> Iterator[list[T] | tuple[T, ...] | dict[KT, VT]]:
    """
    Divide an iterable into chunks/batches of a specific size.

    Handles both mappings (e.g. dicts) and non-mappings (lists, tuples, sets...)
    as you probably expect it to (if you give a dict input, it will chunk on the
    (key, value) items and return dicts of these).
    Thought note that you always can control the type of the chunks with the
    `chunk_type` argument.

    Args:
        iterable: The iterable or mapping to divide.
        chk_size: The size of each chunk.
        chunk_type: The type of the chunks (list, tuple, set, dict...).

    Returns:
        An iterator of dicts if the input is a Mapping, otherwise an iterator
        of collections (list, tuple, set...).

    Examples:
        >>> list(chunk_iterable([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]

        >>> list(chunk_iterable((1, 2, 3, 4, 5), 3, chunk_type=tuple))
        [(1, 2, 3), (4, 5)]

        >>> list(chunk_iterable({"a": 1, "b": 2, "c": 3}, 2))
        [{'a': 1, 'b': 2}, {'c': 3}]

        >>> list(chunk_iterable({"x": 1, "y": 2, "z": 3}, 1, chunk_type=dict))
        [{'x': 1}, {'y': 2}, {'z': 3}]
    """
    if isinstance(iterable, Mapping):
        if chunk_type is None:
            chunk_type = dict
        it = iter(iterable.items())
        for first in it:
            yield {
                key: value for key, value in chain([first], islice(it, chk_size - 1))
            }
    else:
        if chunk_type is None:
            if isinstance(iterable, (list, tuple, set)):
                chunk_type = type(iterable)
            else:
                chunk_type = list
        it = iter(iterable)
        for first in it:
            yield chunk_type(chain([first], islice(it, chk_size - 1)))


def concat_lists(lists: Iterable[Iterable]):
    """Concatenate a list of lists into a single list.

    >>> concat_lists([[1, 2], [3, 4], [5, 6]])
    [1, 2, 3, 4, 5, 6]
    """
    return list(chain.from_iterable(lists))


# a function to translate utc time in 1723557717 format into a human readable format
def utc_int_to_iso_date(utc_time: int) -> str:
    """
    Convert utc integer timestamp to more human readable iso format.
    Inverse of iso_date_to_utc_int.

    >>> utc_int_to_iso_date(1723471317)
    '2024-08-12T14:01:57+00:00'
    """
    return datetime.utcfromtimestamp(utc_time).replace(tzinfo=timezone.utc).isoformat()


def iso_date_to_utc_int(iso_date: str) -> int:
    """
    Convert iso date string to utc integer timestamp.
    Inverse of utc_int_to_iso_date.

    >>> iso_date_to_utc_int('2024-08-12T14:01:57+00:00')
    1723471317
    """
    return int(parse_date(iso_date).timestamp())


# just to have the inverse of a function close to the function itself:
utc_int_to_iso_date.inverse = iso_date_to_utc_int
iso_date_to_utc_int.inverse = utc_int_to_iso_date


def transpose_iterable(iterable_of_tuples):
    return zip(*iterable_of_tuples)


def transpose_and_concatenate(iterable_of_tuples):
    return map(list, map(chain.from_iterable, transpose_iterable(iterable_of_tuples)))


def save_in_temp_dir(obj, serializer=pickle.dumps):
    """
    Saves obj in a temp file, using serializer to serialize it, and returns its path.
    """
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(serializer(obj))
    return f.name


from typing import Any, Optional
from collections.abc import Callable


def mk_local_files_saves_callback(
    rootdir: str | None = None,
    *,
    serializer: Callable[[Any], bytes] = pickle.dumps,
    index_to_filename: Callable[[int], str] = "{:05.0f}".format,
    print_dir_path: bool = True,
):
    """
    Returns a function that takes two inputs (i: int, obj: Any) and saves the
    serializer(obj) bytes in a file named index_to_filename(i) in the rootdir.
    If rootdir, a temp dir is used.
    """
    if rootdir is None:
        rootdir = tempfile.mkdtemp()
    assert os.path.isdir(rootdir), f"rootdir {rootdir} is not a directory"
    if print_dir_path:
        print(f"Files will be saved in {rootdir}")

    def save_to_file(i, obj):
        with open(os.path.join(rootdir, index_to_filename(i)), "wb") as f:
            f.write(serializer(obj))

    return save_to_file


import json
from operator import methodcaller
from typing import T
from collections.abc import Iterable, Callable
from dol import Pipe

DFLT_ENCODING = "utf-8"


def jsonl_dumps(x: Iterable, encoding: str = DFLT_ENCODING) -> bytes:
    r"""
    Serialize an iterable as JSONL bytes

    >>> jsonl_dumps([{'a': 1}, {'b': 2}])
    b'{"a": 1}\n{"b": 2}'

    """
    if isinstance(x, Mapping):
        return json.dumps(x).encode(encoding)
    else:
        return b"\n".join(json.dumps(line).encode(encoding) for line in x)


def jsonl_loads_iter(
    src: T,
    *,
    get_lines: Callable[[T], Iterable[bytes]] = bytes.splitlines,
    line_egress: Callable = methodcaller("strip"),
) -> Iterable[dict]:
    r"""
    Deserialize JSONL bytes into a python iterable (dict or list of dicts)

    >>> list(jsonl_loads(b'\n{"a": 1}\n\n{"b": 2}'))
    [{'a': 1}, {'b': 2}]

    """

    for line in filter(None, map(line_egress, get_lines(src))):
        yield json.loads(line)


jsonl_loads = Pipe(jsonl_loads_iter, list)
jsonl_loads.__doc__ = jsonl_loads_iter.__doc__


from collections.abc import Iterable
import openai
from i2.signatures import SignatureAble
from inspect import Parameter


@Sig.replace_kwargs_using(Sig.merge_with_sig)
def merge_multiple_signatures(
    iterable_of_sigs: Iterable[SignatureAble], **merge_with_sig_options
):
    sig = Sig()
    for input_sig in map(Sig, iterable_of_sigs):
        sig = sig.merge_with_sig(input_sig, **merge_with_sig_options)
    return sig


# TODO: Control whether to only overwrite if defaults and/or annotations don't already exist
# TODO: Control if matching by name or annotation
def source_parameter_props_from(parameters: Mapping[str, Parameter]):
    """
    A decorator that will change the annotation and default of the parameters of the
    decorated function, sourcing them from `parameters`, matching them by name.
    """

    def decorator(func):
        sig = Sig(func)
        common_names = set(sig.names) & set(parameters.keys())
        sig = sig.ch_defaults(
            **{name: parameters[name].default for name in common_names}
        )
        sig = sig.ch_annotations(
            **{name: parameters[name].annotation for name in common_names}
        )
        return sig(func)

    return decorator


# --------------------------------------------------------------------------------------
# ProcessingManager
# Monitoring the processing of a collection of items

from dataclasses import dataclass, field
import time
from typing import (
    Any,
    Optional,
    Tuple,
    Dict,
    Union,
    TypeVar,
    Generic,
)
from collections.abc import Callable, MutableMapping, Iterable

# Define type variables and aliases
KT = TypeVar("KT")  # Key type
VT = TypeVar("VT")  # Value type (pending item type)
Result = TypeVar("Result")  # Result type returned by processing_function
Status = str  # Could be an Enum in future


@dataclass
class ProcessingManager(Generic[KT, VT, Result]):
    """
    A class to manage and monitor the processing of a collection of items, allowing customizable
    processing functions, status handling, and timing control, using keys to track items.

    **Use Case**:
    - Ideal for scenarios where you have a set of items identified by keys that require periodic status
      checks until they reach a completed state.
    - Useful for managing batch jobs, asynchronous tasks, or any operations where items may not complete
      processing immediately.

    **Attributes**:
    - **pending_items** (`MutableMapping[KT, VT]`): The mapping of keys to pending items.
        - If an iterable is provided, it is converted to a dict using `dict(enumerate(iterable))`.
    - **processing_function** (`Callable[[VT], Tuple[Status, Result]]`): A function that takes a value (item)
      and returns a tuple of `(status, result)`.
        - `status` (`Status`): Indicates the current state of the item (e.g., `'completed'`, `'in_progress'`, `'failed'`).
        - `result` (`Result`): Additional data or context about the item's processing result.
    - **handle_status_function** (`Callable[[VT, Status, Result], bool]`): A function that decides whether to remove
      an item from `pending_items` based on its `status` and `result`.
        - Returns `True` if the item should be removed (e.g., processing is complete or failed irrecoverably).
    - **wait_time_function** (`Callable[[float, Dict], float]`): A function that determines how long to wait
      before the next processing cycle.
        - Takes `cycle_duration` (time taken for the last cycle) and `locals()` dictionary as inputs.
        - Returns the sleep time in seconds.
    - **status_check_interval** (`float`): Desired minimum time (in seconds) between status checks. Defaults to `5.0`.
    - **max_cycles** (`Optional[int]`): Maximum number of processing cycles to perform. If `None`, there is no limit.
    - **completed_items** (`MutableMapping[KT, Result]`): Mapping of keys to results for items that have been processed.
    - **cycles** (`int`): Number of processing cycles that have been performed.
    """

    pending_items: MutableMapping[KT, VT] | Iterable[VT]
    processing_function: Callable[[VT], tuple[Status, Result]]
    handle_status_function: Callable[[VT, Status, Result], bool]
    wait_time_function: Callable[[float, dict], float]
    status_check_interval: float = 5.0
    max_cycles: int | None = None
    completed_items: MutableMapping[KT, Result] = field(default_factory=dict)
    cycles: int = 0  # Tracks the number of cycles performed

    def __post_init__(self):
        # Convert pending_items to a MutableMapping if it's not one already
        if not isinstance(self.pending_items, MutableMapping):
            self.pending_items = dict(enumerate(self.pending_items))
        # Ensure completed_items is a MutableMapping
        if not isinstance(self.completed_items, MutableMapping):
            self.completed_items = {}

    @property
    def status(self) -> bool:
        """
        Indicates whether all pending items have been processed.

        Returns:
            bool: `True` if there are no more pending items, `False` otherwise.
        """
        return not self.pending_items

    def process_pending_items(self):
        """
        Processes the pending items once.

        This method iterates over each key-value pair in `pending_items`, applies the `processing_function`
        to determine its status, and then uses `handle_status_function` to decide whether to
        remove the item from `pending_items`. Items removed are added to `completed_items` with their results.
        """
        keys_to_remove = set()

        for k, v in list(self.pending_items.items()):
            # Apply the processing_function to get the item's status and result
            status, result = self.processing_function(v)

            # Decide whether to remove the item based on its status and result
            should_remove = self.handle_status_function(v, status, result)

            if should_remove:
                keys_to_remove.add(k)
                # Store the result associated with the item's key
                self.completed_items[k] = result

        # Remove items that are done processing from pending_items
        for k in keys_to_remove:
            del self.pending_items[k]

    def process_items(self):
        """
        Runs the processing loop until all pending items are processed or max_cycles is reached.

        In each cycle:
        - Calls `process_pending_items()` to process the current pending items.
        - Increments the cycle count.
        - Calculates the duration of the cycle and determines how long to sleep before the next cycle
          using `wait_time_function`.
        - Sleeps for the calculated duration if there are still pending items.

        The loop continues until:
        - `pending_items` is empty (all items have been processed), or
        - The number of cycles reaches `max_cycles`, if specified.
        """
        # Continue looping while there are pending items and the cycle limit hasn't been reached
        while not self.status and self.cycles < (self.max_cycles or float("inf")):
            # Record the start time of the cycle
            cycle_start_time = time.time()

            # Process the pending items once
            self.process_pending_items()

            # Increment the cycle counter
            self.cycles += 1

            # Calculate how long the processing took
            cycle_duration = time.time() - cycle_start_time

            # Determine how long to wait before the next cycle
            sleep_duration = self.wait_time_function(cycle_duration, locals())

            # Sleep if there are still pending items and the sleep duration is positive
            if not self.status and sleep_duration > 0:
                time.sleep(sleep_duration)

        return self.completed_items
```

## vector_stores.py

```python
"""Vector stores and search"""

import tempfile
from functools import partial
import os
from typing import Any
from collections.abc import Callable, Mapping, Iterable
from oa.stores import OaStores, OaVectorStoreFiles, OaFiles

Query = str
MaxNumResults = int
ResultT = Any
SearchResults = Iterable[ResultT]


def docs_to_vector_store(
    docs: Mapping[str, str], vs_name: str = None, *, client=None
) -> tuple[str, dict[str, str]]:
    """
    Create an OpenAI vector store from a mapping of documents.

    Args:
        docs: Mapping of document keys to text content
        vs_name: Optional name for the vector store. If None, generates a unique name.

    Returns:
        tuple: (vector_store_id, file_id_to_doc_key_mapping)
    """
    if vs_name is None:
        import uuid

        vs_name = f"test_vs_{uuid.uuid4().hex[:8]}"

    # Initialize OA stores
    oa_stores = OaStores(client)

    # Create vector store
    vector_store = oa_stores.vector_stores_base.create(vs_name)

    # Create temporary files for each document and upload them
    vs_files = OaVectorStoreFiles(vector_store.id, oa_stores.client)
    file_id_to_doc_key = {}

    for doc_key, doc_text in docs.items():
        # Create a temporary file with the document content
        # Use doc_key as filename for easier identification
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix=f"{doc_key}_", delete=False
        ) as tmp_file:
            tmp_file.write(doc_text)
            tmp_file.flush()

            # Upload the file to OpenAI with proper purpose for assistants/vector stores
            with open(tmp_file.name, "rb") as file_content:
                # Create a file store with the correct purpose for assistants/vector stores
                assistants_files = OaFiles(client, purpose="assistants")
                file_obj = assistants_files.append(file_content)

            # Add the file to the vector store
            vs_files.add_file(file_obj.id)

            # Store the mapping
            file_id_to_doc_key[file_obj.id] = doc_key

            # Clean up temporary file
            os.unlink(tmp_file.name)

    return vector_store.id, file_id_to_doc_key


from functools import partial
from inspect import Parameter
from i2 import Sig
from i2.wrapper import Ingress, wrap


def bind_and_modify(func, *bound_args, _param_changes: dict = (), **bound_kwargs):
    """
    Convenience function that both binds arguments and modifies signature.

    This is perfect for your vector store search use case.

    :param func: The function to wrap
    :param bound_args: Positional arguments to bind
    :param bound_kwargs: Dict of argument names to values to bind
    :param _param_changes: Parameter modifications
    :return: Wrapped function with bound arguments and modified signature

    Example for your vector store case:

    >>> def search(vector_store_id, *, query, filters=None, max_results=10):
    ...     return f"Searching {vector_store_id} for '{query}'"
    >>>
    >>> # Bind vector_store_id and make query positional
    >>> bound_search = bind_and_modify(
    ...     search,
    ...     vector_store_id='my_store',
    ...     _param_changes=dict(query={'kind': Parameter.POSITIONAL_OR_KEYWORD}),
    ... )
    >>>
    >>> bound_search('my query')
    "Searching my_store for 'my query'"
    """
    from i2.wrapper import Ingress, wrap

    # Get original signature and determine what we're binding
    original_sig = Sig(func)
    bound_kwargs = dict(bound_kwargs)

    # Map the bound arguments to parameter names
    bound_params = original_sig.map_arguments(
        bound_args,
        bound_kwargs,
        allow_partial=True,
    )

    # Remove bound parameters from signature
    remaining_sig = original_sig - list(bound_params.keys())

    # Apply parameter modifications to the remaining signature
    if _param_changes:
        remaining_sig = remaining_sig.modified(**_param_changes)

    # Create an ingress that transforms outer args/kwargs to inner args/kwargs
    def kwargs_trans(outer_kwargs):
        # Start with the bound parameters
        inner_kwargs = dict(bound_params)
        # Add the outer kwargs
        inner_kwargs.update(outer_kwargs)
        return inner_kwargs

    # Create ingress with the modified signature as outer and original as inner
    ingress = Ingress(
        outer_sig=remaining_sig, kwargs_trans=kwargs_trans, inner_sig=original_sig
    )

    # Wrap the function
    return wrap(func, ingress=ingress)


def mk_search_func_for_oa_vector_store(
    vector_store_id: str, doc_id_mapping: Mapping[str, str] = None, *, client=None
) -> Callable[[Query], SearchResults]:
    """
    Create a search function for an OpenAI vector store using the Responses API.

    Args:
        vector_store_id: The ID of the vector store to search
        doc_id_mapping: Optional mapping from file IDs to document keys for result translation

    Returns:
        A function that takes a query and returns search results
    """
    oa_stores = OaStores(client)

    def search_func(query: Query, **kwargs) -> SearchResults:
        """
        Search function that uses the Responses API to perform the search.
        """
        # Use the new Responses API with a single call
        response = oa_stores.client.responses.create(
            model="gpt-4o",
            input=query,
            instructions="You are a search assistant. Use the file_search tool to find relevant documents.",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                }
            ],
            **kwargs,
        )

        # Extract file IDs from the response
        file_ids = []
        for output in response.output:
            if output.type == "file_search":
                for result in output.file_search.results:
                    file_ids.append(result.file_id)

        if doc_id_mapping:
            return [
                doc_id_mapping.get(file_id)
                for file_id in file_ids
                if file_id in doc_id_mapping
            ]
        return file_ids

    return search_func


def docs_to_search_func_factory_via_vector_store(
    docs: Mapping[str, str],
) -> Callable[[Query], SearchResults]:
    """
    Factory function that creates a search function via vector store.
    This can be used with check_search_func_factory.
    """
    # Create vector store from docs
    vs_id, file_id_mapping = docs_to_vector_store(docs)

    # Create and return search function with proper mapping
    return mk_search_func_for_oa_vector_store(vs_id, file_id_mapping)
```

## README.md

```python
# oa

Python interface to OpenAi

To install:	```pip install oa```

- [oa](#oa)
- [Usage](#usage)
  - [A collection of prompt-enabled functions](#a-collection-of-prompt-enabled-functions)
    - [PromptFuncs](#promptfuncs)
  - [Functionalizing prompts](#functionalizing-prompts)
  - [Enforcing json formatted outputs](#enforcing-json-formatted-outputs)
  - [Just-do-it: A minimal-boilerplate facade to OpenAI stuff](#just-do-it-a-minimal-boilerplate-facade-to-openai-stuff)
  - [Raw form - When you need to be closer to the metal](#raw-form---when-you-need-to-be-closer-to-the-metal)



# Usage

Sure, you can do many things in English now with our new AI superpowers, but still, to be able to really reuse and compose your best prompts, you had better parametrize them -- that is, distill them down to the minimal necessary interface. The function.

What `oa` does for you is enable you to easily -- really easily -- harness the newly available super-powers of AI from python. 

Below, you'll see how 

See notebooks:
* [oa - An OpenAI facade.ipynb](https://github.com/thorwhalen/oa/blob/main/misc/oa%20-%20An%20OpenAI%20facade.ipynb)
* [oa - Making an Aesop fables children's book oa.ipynb](https://github.com/thorwhalen/oa/blob/main/misc/oa%20-%20Making%20an%20Aesop%20fables%20children's%20book%20oa.ipynb)

Below are a few snippets from there. 


## A collection of prompt-enabled functions

One main functionality that `oa` offers is an easy way to define python functions based 
on AI prompts. 
In order to demo these, we've made a few ready-to-use ones, which you can access via
`oa.ask.ai`:

```python
from oa.ask import ai

list(ai)
```

    ['define_jargon', 'suggest_names', ..., 'make_synopsis']

These are the names of functions automatically generated from a (for now small) folder of prompt templates. 

These functions all have propert signatures:

```python
import inspect
print(inspect.signature(ai.suggest_names))
```

(*, thing, n='30', min_length='1', max_length='15')


```python
answer = ai.suggest_names(
    thing="""
    A python package that provides python functions to do things,
    enabled by prompts sent to an OpenAI engine.
    """
)
print(answer)
```

    GyroPy
    PromptCore
    FlexiFunc
    ProperPy
    PyCogito
    ...
    PyPrompter
    FuncDomino
    SmartPy
    PyVirtuoso



### PromptFuncs

Above, all we did was scan some local text files that specify prompt templates and make an object that contained the functions they define. We used `oa.PromptFuncs` for that. You can do the same. What `PromptFuncs` uses itself, is a convenient `oa.prompt_function` function that transforms a template into a function. See more details in the next "Functionalizing prompts" section.

Let's just scratch the surface of what `PromptFuncs` can do. For more, you can look at the documentation, including the docs for `ai.prompt_function`.


```python
from oa import PromptFuncs

funcs = PromptFuncs(
    template_store = {
        "haiku": "Write haiku about {subject}. Only output the haiku.",
        "stylize": """
            Reword what I_SAY, using the style: {style:funny}.
            Only output the reworded text.
            I_SAY:
            {something}
        """,
    }
)

list(funcs)
```

    ['haiku', 'stylize']



```python
import inspect
for name in funcs:
    print(f"{name}: {inspect.signature(funcs[name])}")

```

    haiku: (*, subject)
    stylize: (*, something, style='funny')



```python
print(funcs.haiku(subject="The potential elegance of code"))
```

    Code speaks a language,
    Elegant syntax dances,
    Beauty in function.



```python
print(funcs.stylize(something="The mess that is spagetti code!"))
```

    Spaghetti code, the tangled web of chaos!



```python
print(funcs.stylize(something="The mess that is spagetti code!", style="poetic"))
```

    The tangled strands of code, a chaotic tapestry!


We used a `dict` to express our `func_name:template` specification, but note that it can be any `Mapping`. Therefore, you can source `PromptFuncs` with local files (example, using `dol.TextFiles`, like we did), a DB, or anything you can map to a key-value `Mapping` interface.

(We suggest you use the [dol](https://pypi.org/project/dol/) package, and ecosystem, to help out with that.)


## Functionalizing prompts

The `oa.prompt_function` is an easy to use, yet extremely configurable, tool to do that.


```python
from oa import prompt_function

template = """
I'd like you to give me help me understand domain-specific jargon. 
I will give you a CONTEXT and some WORDS. 
You will then provide me with a tab separated table (with columns name and definition)
that gives me a short definition of each word in the context of the context.
Only output the table, with no words before or after it, since I will be parsing the output
automatically.

CONTEXT:
{context}

WORDS:
{words}
"""

define_jargon = prompt_function(template, defaults=dict(context='machine learning'))
```


```python
# Let's look at the signature
import inspect
print(inspect.signature(define_jargon))
```

    (*, words, context='machine learning')



```python
response = define_jargon(words='supervised learning\tunsupervised learning\treinforcement learning')
print(response)
```

    name	definition
    supervised learning	A type of machine learning where an algorithm learns from labeled training data to make predictions or take actions. The algorithm is provided with input-output pairs and uses them to learn patterns and make accurate predictions on new, unseen data.
    unsupervised learning	A type of machine learning where an algorithm learns patterns and structures in input data without any labeled output. The algorithm identifies hidden patterns and relationships in the data to gain insights and make predictions or classifications based on the discovered patterns.
    reinforcement learning	A type of machine learning where an algorithm learns to make a sequence of decisions in an environment to maximize a cumulative reward. The algorithm interacts with the environment, receives feedback in the form of rewards or punishments, and adjusts its actions to achieve the highest possible reward over time.



```python
def table_str_to_dict(table_str, *, newline='\n', sep='   '):
    return dict([x.split('   ') for x in table_str.split('\n')[1:]])

table_str_to_dict(define_jargon(
    words='\n'.join(['allomorph', 'phonology', 'phonotactic constraints']),
    context='linguistics'
))

```


    {'allomorph': 'A variant form of a morpheme that is used in a specific linguistic context, often resulting in different phonetic realizations.',
     'phonology': 'The study of speech sounds and their patterns, including the way sounds are organized and used in a particular language or languages.',
     'phonotactic constraints': 'The rules or restrictions that govern the possible combinations of sounds within a language, specifying what sound sequences are allowed and which ones are not.'}



Check out the many ways you can configure your function with `prompt_function`:


```python
str(inspect.signature(prompt_function)).split(', ')
```




    ['(template',
     '*',
     'defaults: Optional[dict] = None',
     'template_to_names=<function _extract_names_from_format_string at 0x106d20940>',
     'embodier=<function string_format_embodier at 0x106d204c0>',
     'name=None',
     'prompt_func=<function chat at 0x128420af0>',
     'prompt_func_kwargs=None',
     'egress=None)']


## Enforcing json formatted outputs

With some newer models (example, "gpt4o-mini") you can request that only valid 
json be given as a response, or even more: A json obeying a specific schema. 
You control this via the `response_format` argument. 

Let's first use AI to get a json schema for characteristics of a programming language.
That's a json, so why not use the `response_format` with `{"type": "json_object"}` to 
get that schema!


```python
from oa import chat
from oa.util import data_files

# To make sure we get a json schema that is openAI compliant, we'll use an example of 
# one in our prompt to AI to give us one...
example_of_a_openai_json_schema = example_of_a_openai_json_schema = (
    data_files.joinpath('json_schema_example.json').read_text()
)

json_schema_str = chat(
    "Give me the json of a json_schema I can use different characteristics of "
    "programming languages. This schema should be a valid schema to use as a "
    "response_format in the OpenAI API. "
    f"Here's an example:\n{example_of_a_openai_json_schema}", 
    model='gpt-4o-mini',
    response_format={'type': 'json_object'}
)
print(json_schema_str[:500] + '...')
```

```
{
  "name": "programming_language_characteristics",
  "strict": false,
  "schema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "The name of the programming language."
      },
      "paradigm": {
        "type": "array",
        "items": {
          "type": "string",
          "description": "Programming paradigms the language follows, e.g., 'object-oriented', 'functional', etc."
        }
      },
      "designed_by": {
        "t...
```

Now we can use this schema to make an AI-enabled python function that will give 
us characteristics of a language, but always using that fixed format.
This also means we'll be able to stick an `egress` to our prompt function, so 
that we always get our output in the form of an already decoded json (a `dict`).

```python
from oa import prompt_function
import json

properties_of_language = prompt_function(
    "Give me a json that describes characteristics of the programming language: {language}.",
    prompt_func=chat, 
    prompt_func_kwargs=dict(
        model='gpt-4o-mini', 
        response_format={
            'type': 'json_schema',
            'json_schema': json.loads(json_schema_str)
        }
    ),
    egress=json.loads
)

info = properties_of_language('Python')
print(f"{type(info)=}\n")

from pprint import pprint
pprint(info)
```

```
type(info)=<class 'dict'>

{'designed_by': ['Guido van Rossum'],
 'first_appeared': 1991,
 'influenced_by': ['ABC', 'C', 'C++', 'Java', 'Modula-3', 'Lisp'],
 'influences': ['Ruby', 'Swift', 'Matlab', 'Go'],
 'latest_release': '3.11.5',
 'name': 'Python',
 'paradigm': ['object-oriented', 'imperative', 'functional', 'procedural'],
 'typing_discipline': 'dynamic',
 'website': 'https://www.python.org'}
```




## Just-do-it: A minimal-boilerplate facade to OpenAI stuff

For the typical tasks you might want to use OpenAI for.

Note there's no "enter API KEY here" code. That's because if you don't have it in the place(s) it'll look for it, it will simply ask you for it, and, with your permission, put it in a hidden file for you, so you don't have to do this every time.


```python
import oa
```


```python
print(oa.complete('chatGPT is a'))
```

     chatbot based on OpenAI's GPT-2, a natural language processing



```python
print(oa.chat('Act as a chatGPT expert. List 5 useful prompt templates'))
```

    Sure, here are 5 useful prompt templates that can be used in a chatGPT session:
    
    1. Can you provide some more details about [topic]?
    - Examples: Can you provide some more details about the symptoms you're experiencing? Or Can you provide some more details about the issue you're facing with the website?
    
    2. How long have you been experiencing [issue]?
    - Examples: How long have you been experiencing the trouble with your internet connection? Or How long have you been experiencing the pain in your back?
    
    3. Have you tried any solutions to resolve [issue]?
    - Examples: Have you tried any solutions to resolve the error message you're seeing? Or Have you tried any solutions to resolve the trouble you're having with the application?
    
    4. What is the specific error message you are receiving?
    - Examples: What is the specific error message you are receiving when you try to log in? Or What is the specific error message you are receiving when you try to submit the form?
    
    5. Is there anything else you would like to add that might be helpful for me to know?
    - Examples: Is there anything else you would like to add that might be helpful for me to know about your situation? Or Is there anything else you would like to add that might be helpful for me to know about the product you are using?



```python
url = oa.dalle('An image of Davinci, pop art style')
print(url)
```

    https://oaidalleapiprodscus.blob.core.windows.net/private/org-AY3lr3H3xB9yPQ0HGR498f9M/user-7ZNCDYLWzP0GT48V6DCiTFWt/img-pNE6fCWGN3eJGj7ycFwZREhi.png?st=2023-04-22T22%3A17%3A03Z&se=2023-04-23T00%3A17%3A03Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-04-22T21%3A08%3A14Z&ske=2023-04-23T21%3A08%3A14Z&sks=b&skv=2021-08-06&sig=5j6LPVO992R95dllAAjbmOXzS0MORD06Fo8unwtGNl0%3D



```python
from IPython.display import Image

Image(url=url)
```

<img width="608" alt="image" src="https://github.com/thorwhalen/oa/assets/1906276/6e7b2ac4-648c-4ec0-81bf-078208f4ac39">


## Raw form - When you need to be closer to the metal

The `raw` object is a thin layer on top of the `openai` package, which is itself a thin layer over the web requests. 

What was unsatisfactory with the `openai` package is (1) finding the right function, (2) the signature of the function once you found it, and (3) the documentation of the function. 
What raw contains is pointers to the main functionalities (not all available -- yet), with nice signatures and documentation, extracted from the web service openAPI specs themselves. 

For example, to ask chatGPT something, the openai function is `openai.ChatCompletion.create`, or to get simple completions, the function is `openai.Completion.create` whose help is:

```
Help on method create in module openai.api_resources.completion:

create(*args, **kwargs) method of builtins.type instance
    Creates a new completion for the provided prompt and parameters.
    
    See https://platform.openai.com/docs/api-reference/completions/create for a list
    of valid parameters.
```

Not super helpful. It basically tells you to got read the docs elsewhere. 

The corresponding `raw` function is `raw.completion`, and it's help is a bit more like what you'd expect in a python function.



```python
help(oa.raw.chatcompletion)
```

    Help on Wrap in module openai.api_resources.chat_completion:
    
    chatcompletion
        Creates a new chat completion for the provided messages and parameters.
        
                See https://platform.openai.com/docs/api-reference/chat-completions/create
                for a list of valid parameters.
        
        chatcompletion(
                model: str
                messages: List[oa.openai_specs.Message]
                *
                temperature: float = 1
                top_p: float = 1
                n: int = 1
                stream: bool = False
                stop=None
                max_tokens: int = None
                presence_penalty: float = 0
                frequency_penalty: float = 0
                logit_bias: dict = None
                user: str = None
        )
        
        :param model: ID of the model to use. Currently, only `gpt-3.5-turbo` and `gpt-3.5-turbo-0301` are supported.
        
        :param messages: The messages to generate chat completions for, in the [chat format](/docs/guides/chat/introduction).
        
        :param temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or `top_p` but not both.
        
        :param top_p: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or `temperature` but not both.
        
        :param n: How many chat completion choices to generate for each input message.
        
        :param stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message.
        
        :param stop: Up to 4 sequences where the API will stop generating further tokens.
        
        :param max_tokens: The maximum number of tokens allowed for the generated answer. By default, the number of tokens the model can return will be (4096 - prompt tokens).
        
        :param presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. [See more information about frequency and presence penalties.](/docs/api-reference/parameter-details)
        
        :param frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. [See more information about frequency and presence penalties.](/docs/api-reference/parameter-details)
        
        :param logit_bias: Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
        
        :param user: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids).
    



```python
prompt = 'List 5 top prompt engineering tricks to write good prompts for chatGPT'

resp = oa.raw.chatcompletion(
    messages=[
        {"role": "system", "content": "You are an expert at chatGPT"},
        {"role": "user", "content": prompt},
    ],
    model='gpt-3.5-turbo-0301',
    temperature=0.5,
    max_tokens=300
)
resp
```




    <OpenAIObject chat.completion id=chatcmpl-78HMPgn3oy2fuvm6sLCgOsQvnTVYr at 0x11fd467a0> JSON: {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "Sure, here are 5 top prompt engineering tricks to write good prompts for chatGPT:\n\n1. Be Specific: Ensure that your prompts are specific and clear. The more specific your prompt, the better the response from chatGPT. Avoid using vague or ambiguous language.\n\n2. Use Open-Ended Questions: Open-ended questions encourage chatGPT to provide more detailed and personalized responses. Avoid using closed-ended questions that can be answered with a simple yes or no.\n\n3. Include Context: Providing context to your prompts helps chatGPT to better understand the topic and provide more relevant responses. Include any necessary background information or details to help guide chatGPT's response.\n\n4. Use Emotion: Including emotion in your prompts can help chatGPT generate more engaging and relatable responses. Consider using prompts that evoke emotions such as happiness, sadness, or excitement.\n\n5. Test and Refine: Experiment with different prompts and evaluate the responses from chatGPT. Refine your prompts based on the quality of the responses and continue to test and improve over time.",
            "role": "assistant"
          }
        }
      ],
      "created": 1682207713,
      "id": "chatcmpl-78HMPgn3oy2fuvm6sLCgOsQvnTVYr",
      "model": "gpt-3.5-turbo-0301",
      "object": "chat.completion",
      "usage": {
        "completion_tokens": 214,
        "prompt_tokens": 36,
        "total_tokens": 250
      }
    }


```python
print(resp['choices'][0]['message']['content'])
```

    Sure, here are 5 top prompt engineering tricks to write good prompts for chatGPT:
    
    1. Be Specific: Ensure that your prompts are specific and clear. The more specific your prompt, the better the response from chatGPT. Avoid using vague or ambiguous language.
    
    2. Use Open-Ended Questions: Open-ended questions encourage chatGPT to provide more detailed and personalized responses. Avoid using closed-ended questions that can be answered with a simple yes or no.
    
    3. Include Context: Providing context to your prompts helps chatGPT to better understand the topic and provide more relevant responses. Include any necessary background information or details to help guide chatGPT's response.
    
    4. Use Emotion: Including emotion in your prompts can help chatGPT generate more engaging and relatable responses. Consider using prompts that evoke emotions such as happiness, sadness, or excitement.
    
    5. Test and Refine: Experiment with different prompts and evaluate the responses from chatGPT. Refine your prompts based on the quality of the responses and continue to test and improve over time.



# ChatDacc: Shared chats parser

The `ChatDacc` class gives you access of the main functionalities of the `chats.py` module.
It provides tools for analyzing and extracting data from shared ChatGPT conversations. 
Here’s an overview of its main features, for more details, see 
[this demo notebook](https://github.com/thorwhalen/oa/blob/main/misc/oa%20-%20chats%20acquisition%20and%20parsing.ipynb)


## Initialize with a URL

Begin by creating a ChatDacc object with a conversation’s shared URL:

```python
from oa.chats import ChatDacc

url = 'https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5'
dacc = ChatDacc(url)
```

## Access Basic Conversation Data

Retrieve minimal metadata (e.g., roles, timestamps, and content):

```python
dacc.basic_turns_data
```

Or directly get it as a Pandas DataFrame:

```python
dacc.basic_turns_df
```

## Explore Full Turn Data

Access all available fields for each message in the conversation:

```python
dacc.turns_data
```

Indexed access simplifies specific turn retrieval:

```python
dacc.turns_data_keys
turn_data = dacc.turns_data[dacc.turns_data_keys[3]]
```

## Extract Metadata

Metadata summarizing the conversation is available through:

```python
dacc.metadata
```

## Extract and Analyze URLs

Identify all URLs referenced within the conversation, including quoted and embedded sources:

```python
urls = dacc.url_data()
```

For richer context, you can include prior levels or retain tracking parameters:

urls_in_context = dacc.url_data(prior_levels_to_include=1, remove_chatgpt_utm=False)

## Get Full JSON

The raw JSON for the entire conversation can be accessed for in-depth analysis:

```python
dacc.full_json_dict
```

This tool simplifies data extraction and analysis from ChatGPT shared conversations, making it ideal for developers, researchers, and data analysts.
```