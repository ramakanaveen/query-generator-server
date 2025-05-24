# app/services/ai_description_service.py

from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime

from app.core.db import db_pool
from app.core.logging import logger
from app.services.llm_provider import LLMProvider
from app.services.query_generation.prompts.ai_description_prompts import column_description_prompt, \
    table_description_prompt, examples_prompt


class AISchemaService:
    """Service for AI-assisted schema editing and description generation."""

    def __init__(self):
        self.llm_provider = LLMProvider()

    async def generate_descriptions(
            self,
            column_name: Optional[str] = None,
            table_name: Optional[str] = None,
            schema_name: Optional[str] = None,
            table_context: Optional[Dict[str, Any]] = None,
            count: int = 3
    ) -> List[str]:
        """
        Generate descriptions for tables or columns using AI with comprehensive context.

        Args:
            column_name: Name of the column (for column descriptions)
            table_name: Name of the table
            schema_name: Name of the schema
            table_context: Full table definition for context
            count: Number of description alternatives to generate

        Returns:
            List of description strings
        """
        description_type = "column" if column_name else "table"

        # Get table context from database if not provided
        if not table_context and table_name:
            table_context = await self._fetch_table_context(table_name, column_name)

        # Format table columns for the prompt
        formatted_columns = self._format_columns(table_context)

        # Get LLM model
        llm = self.llm_provider.get_model("gemini")

        # Generate prompt based on type
        if description_type == "column":
            prompt = await self._create_column_description_prompt(
                column_name, table_name, schema_name, table_context,
                formatted_columns, count
            )
        else:
            prompt = await self._create_table_description_prompt(
                table_name, schema_name, formatted_columns, count
            )

        # Send the prompt to the LLM
        response = await llm.ainvoke(prompt)
        response_text = response.content.strip()

        # Parse the response to extract descriptions
        descriptions = self._parse_descriptions(response_text, count)

        return descriptions

    async def generate_examples(
            self,
            table_name: str,
            table_schema: Dict[str, Any],
            count: int = 3
    ) -> List[Dict[str, str]]:
        """
        Generate example queries for a table using AI.

        Args:
            table_name: Name of the table
            table_schema: Schema information for the table
            count: Number of examples to generate

        Returns:
            List of examples with natural language and query
        """
        # Format column information
        columns_text = ""
        if "columns" in table_schema:
            for column in table_schema["columns"]:
                name = column.get("name", "unknown")
                col_type = column.get("type", column.get("kdb_type", "unknown"))
                desc = column.get("description", column.get("column_desc", ""))
                columns_text += f"- {name} ({col_type}): {desc}\n"

        # Get LLM model (use Claude for code generation)
        llm = self.llm_provider.get_model("gemini")
        prompt = await self.create_examples_prompt(table_name, table_schema, columns_text,count)
        # Get response from LLM
        response = await llm.ainvoke(prompt)
        response_text = response.content.strip()

        # Parse the response
        examples = self._parse_examples(response_text, count)

        return examples

    async def _fetch_table_context(
            self,
            table_name: str,
            column_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch table context information from the database."""
        table_context = {}

        try:
            conn = await db_pool.get_connection()
            try:
                # Find the table in any active schema version
                table_query = """
                              SELECT
                                  td.content,
                                  td.description as table_description,
                                  sd.name as schema_name
                              FROM
                                  table_definitions td
                                      JOIN
                                  schema_versions sv ON td.schema_version_id = sv.id
                                      JOIN
                                  schema_definitions sd ON sv.schema_id = sd.id
                                      JOIN
                                  active_schemas a ON sv.id = a.current_version_id
                              WHERE
                                  td.name = $1
                                  LIMIT 1 \
                              """
                table_data = await conn.fetchrow(table_query, table_name)

                if table_data:
                    # Parse content
                    content = table_data["content"]
                    if isinstance(content, str):
                        content = json.loads(content)

                    # Build table context
                    table_context = {
                        "name": table_name,
                        "description": table_data["table_description"],
                        "schema_name": table_data["schema_name"],
                        "columns": content.get("columns", [])
                    }

                    # If looking for a specific column, find it
                    if column_name and "columns" in content:
                        target_column = None
                        for col in content["columns"]:
                            if col.get("name") == column_name:
                                target_column = col
                                break

                        if target_column:
                            # Add specific column type information
                            table_context["target_column"] = target_column
            finally:
                await db_pool.release_connection(conn)
        except Exception as e:
            logger.warning(f"Error fetching table context: {str(e)}")
            # Continue without context

        return table_context

    def _format_columns(self, table_context: Optional[Dict[str, Any]]) -> str:
        """Format table columns for inclusion in prompts."""
        if not table_context or "columns" not in table_context:
            return ""

        formatted_columns = ""
        for col in table_context.get("columns", []):
            col_name = col.get("name", "unknown")
            col_type = col.get("type", col.get("kdb_type", "unknown"))
            col_desc = col.get("description", col.get("column_desc", ""))
            formatted_columns += f"- {col_name} ({col_type}): {col_desc}\n"

        return formatted_columns

    async def _create_column_description_prompt(
            self,
            column_name: str,
            table_name: str,
            schema_name: Optional[str],
            table_context: Optional[Dict[str, Any]],
            formatted_columns: str,
            count: int
    ) -> str:
        """Create a prompt for column description generation."""
        # Get column type information
        column_type = "unknown"
        if table_context and "target_column" in table_context:
            column_type = table_context["target_column"].get("type",
                                                             table_context["target_column"].get("kdb_type", "unknown"))
        elif table_context and "columns" in table_context:
            for col in table_context["columns"]:
                if col.get("name") == column_name:
                    column_type = col.get("type", col.get("kdb_type", "unknown"))
                    break

        return column_description_prompt.format(
            count = count,
            column_name=column_name,
            table_name=table_name,
            schema_name=schema_name or "Unknown",
            table_description=table_context.get("description", "Not provided") if table_context else "Not provided",
            column_type=column_type,
            formatted_columns=formatted_columns or "No columns information available"
        )

    async def _create_table_description_prompt(
            self,
            table_name: str,
            schema_name: Optional[str],
            formatted_columns: str,
            count: int
    ) -> str:
        """Create a prompt for table description generation."""
        return table_description_prompt.format(
            count=count,
            table_name=table_name,
            schema_name=schema_name or "Unknown",
            formatted_columns=formatted_columns or "No columns information available"
        )

    async def create_examples_prompt(self, table_name: str, table_schema: Dict[str, Any], columns_text: str, count: int) -> str:
        """Create a prompt for example generation."""
        return examples_prompt.format(
            table_name=table_name,
            table_description=table_schema.get('description', 'No description provided'),
            columns_text=columns_text,
            count=count
        )

    def _parse_descriptions(self, response_text: str, count: int) -> List[str]:
        """Parse descriptions from LLM response."""
        try:
            # Try to extract JSON array if wrapped in text
            import re
            json_match = re.search(r'\[(.*)\]', response_text, re.DOTALL)
            if json_match:
                json_str = f"[{json_match.group(1)}]"
                descriptions = json.loads(json_str)
            else:
                descriptions = json.loads(response_text)
        except (json.JSONDecodeError, AttributeError):
            # If not valid JSON, try to split by lines
            descriptions = [line.strip() for line in response_text.split('\n') if line.strip()]

            # Remove any leading numbers (in case of formatted lists)
            descriptions = [re.sub(r'^\d+\.\s*', '', desc) for desc in descriptions]

            # Remove quotes if present
            descriptions = [re.sub(r'^["\'](.*)["\']$', r'\1', desc) for desc in descriptions]

        # Ensure we have a list
        if not isinstance(descriptions, list):
            descriptions = [descriptions]

        # Limit to requested count
        return descriptions[:count]

    def _parse_examples(self, response_text: str, count: int) -> List[Dict[str, str]]:
        """Parse examples from LLM response."""
        try:
            # Try to extract JSON array if wrapped in text
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                examples = json.loads(json_match.group(0))
            else:
                examples = json.loads(response_text)
        except (json.JSONDecodeError, AttributeError):
            # If we can't parse as JSON, try manual extraction
            examples = []

            # Look for patterns like "Natural Language:" followed by "Query:"
            nl_query_pairs = re.findall(
                r'(?:Natural Language|Description):\s*(.*?)(?:\n|$).*?(?:Query|KDB/q query):\s*(.*?)(?:\n\n|\Z)',
                response_text,
                re.DOTALL
            )

            for nl, query in nl_query_pairs:
                # Clean up the query (remove ```q and ``` if present)
                clean_query = re.sub(r'```q?\n(.*?)\n```', r'\1', query.strip(), flags=re.DOTALL)

                examples.append({
                    "natural_language": nl.strip(),
                    "query": clean_query.strip()
                })

        # Ensure we have a list
        if not isinstance(examples, list):
            examples = [examples]

        # Limit to requested count
        return examples[:count]

    def generate_fallback_descriptions(self, column_name: str, table_name: str) -> List[str]:
        """Generate fallback descriptions for common column types."""
        column_name = column_name.lower()

        # Common financial data columns
        if any(x in column_name for x in ["date", "dt"]):
            return [
                f"Date when the {table_name} entry was recorded.",
                f"Calendar date for this {table_name} record.",
                f"Trading date associated with this {table_name} data point."
            ]
        elif any(x in column_name for x in ["time", "tm"]):
            return [
                f"Time when the {table_name} event occurred.",
                f"Transaction time for this {table_name} record.",
                f"Time of day when this {table_name} data was captured."
            ]
        elif any(x in column_name for x in ["sym", "symbol", "ticker"]):
            return [
                f"Ticker symbol identifying the financial instrument in this {table_name} record.",
                f"Symbol code representing the security in the {table_name}.",
                f"Instrument identifier for this {table_name} entry."
            ]
        elif any(x in column_name for x in ["price", "px"]):
            return [
                f"Price of the instrument in the {table_name} record.",
                f"Market price recorded for this {table_name} entry.",
                f"Trading price captured in this {table_name} data point."
            ]
        elif any(x in column_name for x in ["vol", "volume", "size", "qty", "quantity"]):
            return [
                f"Trading volume or size recorded in this {table_name} entry.",
                f"Number of units involved in this {table_name} transaction.",
                f"Transaction size captured in the {table_name} record."
            ]
        elif "bid" in column_name:
            return [
                f"Bid price offered by buyers in this {table_name} record.",
                f"The highest price buyers are willing to pay in this {table_name} entry.",
                f"Buy-side quote price in the {table_name}."
            ]
        elif any(x in column_name for x in ["ask", "offer"]):
            return [
                f"Ask price offered by sellers in this {table_name} record.",
                f"The lowest price sellers are willing to accept in this {table_name} entry.",
                f"Sell-side quote price in the {table_name}."
            ]
        elif "mid" in column_name:
            return [
                f"Mid-price between bid and ask for this {table_name} record.",
                f"Average of bid and ask prices in this {table_name} entry.",
                f"Midpoint market price in the {table_name}."
            ]
        elif "side" in column_name:
            return [
                f"Trading side (Buy or Sell) for this {table_name} record.",
                f"Direction of the transaction in this {table_name} entry.",
                f"Indicates whether this {table_name} record represents a buy or sell order."
            ]

        # Generic fallback
        return [
            f"{column_name} data for the {table_name} record.",
            f"{column_name} information stored in the {table_name}.",
            f"{column_name} value associated with this {table_name} entry."
        ]