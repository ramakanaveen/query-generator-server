# app/services/query_generation/nodes/enhanced_schema_description_node.py

from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from app.core.logging import logger
from app.core.profiling import timeit
from app.services.query_generation.prompts.schema_description_prompts import SCHEMA_DESCRIPTION_PROMPT_TEMPLATE

# Import the enhanced service
from app.services.enhanced_schema_service import EnhancedSchemaService, SchemaDescriptionConfig

# Singleton pattern for service
_enhanced_schema_service = None

def get_enhanced_schema_service():
    """Get singleton instance of enhanced schema service."""
    global _enhanced_schema_service
    if _enhanced_schema_service is None:
        _enhanced_schema_service = EnhancedSchemaService()
    return _enhanced_schema_service

@timeit
async def generate_enhanced_schema_description(state):
    """
    Generate schema description using the enhanced schema service.

    Benefits over old approach:
    - Uses optimized database queries
    - Better caching
    - Consistent with query generation architecture
    - Better error handling
    - Performance monitoring
    """
    try:
        # Validate schema targets
        if not hasattr(state, 'schema_targets') or state.schema_targets is None:
            state.schema_targets = {
                "tables": state.directives if hasattr(state, 'directives') else ["*ALL*"],
                "columns": [],
                "detail_level": "standard"
            }
            state.thinking.append("⚠️ No schema targets found, using directives as default")

        schema_targets = state.schema_targets
        llm = state.llm

        # Enhanced thinking step with more detail
        target_tables = schema_targets.get("tables", [])
        target_columns = schema_targets.get("columns", [])
        detail_level = schema_targets.get("detail_level", "standard")

        state.thinking.append("📊 Generating enhanced schema description...")
        state.thinking.append(f"🎯 Targets: {len(target_tables)} tables, {len(target_columns)} columns")
        state.thinking.append(f"📋 Detail level: {detail_level}")

        # Get enhanced schema service
        schema_service = get_enhanced_schema_service()

        # Create configuration based on detail level
        config = SchemaDescriptionConfig(
            detail_level=detail_level,
            include_examples=(detail_level == "detailed"),
            include_relationships=True,
            max_tables=50 if "*ALL*" in target_tables else 20,
            max_columns_per_table=100 if detail_level == "detailed" else 50
        )

        # ✅ KEY IMPROVEMENT: Use enhanced service instead of direct SchemaManager calls
        result = await schema_service.retrieve_schema_for_description(
            schema_targets=schema_targets,
            user_id=getattr(state, 'user_id', None),
            config=config
        )

        # Check if we got results
        schema_data = result.schema_structure
        if not schema_data or not schema_data.get("schemas"):
            state.thinking.append("❌ No schema information found with enhanced service")
            state.generated_content = "I couldn't find any schema information matching your request."
            return state

        # Enhanced logging with performance metrics
        schemas_found = len(schema_data.get("schemas", {}))
        total_tables = sum(
            len(group_data.get("schemas", {}).get(schema_name, {}).get("tables", {}))
            for group_data in schema_data.get("schemas", {}).values()
            for schema_name in group_data.get("schemas", {}).keys()
        )

        state.thinking.append(f"✅ Enhanced retrieval: {schemas_found} schema groups, {total_tables} tables")

        # Log performance metrics
        perf_metrics = result.performance_metrics
        retrieval_time = perf_metrics.get("total_time_ms", 0)
        state.thinking.append(f"⚡ Retrieval time: {retrieval_time:.1f}ms")

        if result.metadata.get("cache_hit"):
            state.thinking.append("💾 Used cached results for optimal performance")

        # Format schema for LLM prompt (enhanced formatting)
        formatted_schema = format_enhanced_schema_for_prompt(schema_data, detail_level)

        # Generate description using LLM
        prompt = ChatPromptTemplate.from_template(SCHEMA_DESCRIPTION_PROMPT_TEMPLATE)
        chain = prompt | llm

        response = await chain.ainvoke({
            "schema_info": formatted_schema,
            "detail_level": detail_level,
            "database_type": state.database_type,
            "query": state.query
        })

        # Update state with results
        state.generated_content = response.content.strip()
        state.schema_description_metadata = {
            "schemas_processed": schemas_found,
            "tables_processed": total_tables,
            "retrieval_time_ms": retrieval_time,
            "cache_hit": result.metadata.get("cache_hit", False),
            "detail_level": detail_level
        }

        state.thinking.append("✅ Generated enhanced schema description")

        return state

    except Exception as e:
        logger.error(f"Error in enhanced schema description: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Enhanced schema description error: {str(e)}")

        # Graceful fallback
        state.generated_content = f"I encountered an error while retrieving schema information: {str(e)}"

        return state

def format_enhanced_schema_for_prompt(schema_data: Dict[str, Any], detail_level: str) -> str:
    """
    Enhanced formatting for schema data with hierarchical structure.

    Better than the old approach because:
    - Handles group > schema > table hierarchy
    - Better markdown formatting
    - Respects detail levels properly
    - More readable output
    """
    if not schema_data or not schema_data.get("schemas"):
        return "No schema information available."

    result = []
    schemas_dict = schema_data.get("schemas", {})

    # Count totals for summary
    total_groups = len(schemas_dict)
    total_schemas = sum(len(group_data.get("schemas", {})) for group_data in schemas_dict.values())
    total_tables = sum(
        len(schema_info.get("tables", {}))
        for group_data in schemas_dict.values()
        for schema_info in group_data.get("schemas", {}).values()
    )

    # Add comprehensive header
    result.append(f"# Database Schema Information\n")
    result.append(f"**Overview**: {total_tables} tables across {total_schemas} schemas in {total_groups} groups\n")

    # Process each group
    for group_name, group_data in schemas_dict.items():
        result.append(f"## Group: {group_name}\n")

        if group_data.get("description"):
            result.append(f"**Description**: {group_data['description']}\n")

        schemas = group_data.get("schemas", {})
        if schemas:
            result.append(f"**Schemas in this group**: {len(schemas)}\n")

            # Process each schema in the group
            for schema_name, schema_info in schemas.items():
                result.append(f"### Schema: {schema_name}\n")

                if schema_info.get("description"):
                    result.append(f"**Description**: {schema_info['description']}\n")

                tables = schema_info.get("tables", {})
                if tables:
                    result.append(f"**Tables**: {len(tables)}\n")

                    # Handle different detail levels
                    if len(tables) > 10 and detail_level == "summary":
                        # For many tables with summary detail, just list them
                        table_names = list(tables.keys())
                        result.append("**Available tables**: " + ", ".join(f"`{name}`" for name in table_names) + "\n")
                    else:
                        # Process each table in detail
                        for table_name, table_data in tables.items():
                            result.append(f"#### Table: `{table_name}`\n")

                            if table_data.get("description"):
                                result.append(f"**Description**: {table_data['description']}\n")

                            columns = table_data.get("columns", [])
                            if columns:
                                result.append(f"**Columns** ({len(columns)} total):\n")

                                if detail_level == "summary":
                                    # Summary: just column names
                                    col_names = [col.get("name", "unknown") for col in columns[:20]]
                                    result.append("- " + ", ".join(f"`{name}`" for name in col_names))
                                    if len(columns) > 20:
                                        result.append(f" ... and {len(columns) - 20} more")
                                    result.append("\n")
                                else:
                                    # Standard/Detailed: full table format
                                    result.append("| Column Name | Data Type | Description |")
                                    result.append("|-------------|-----------|-------------|")

                                    max_cols = 50 if detail_level == "detailed" else 20
                                    for col in columns[:max_cols]:
                                        col_name = col.get("name", "unknown")
                                        col_type = col.get("type", col.get("kdb_type", "unknown"))
                                        col_desc = col.get("description", col.get("column_desc", ""))

                                        # Escape pipes in description
                                        col_desc = col_desc.replace("|", "\\|")

                                        result.append(f"| `{col_name}` | {col_type} | {col_desc} |")

                                    if len(columns) > max_cols:
                                        result.append(f"| ... | ... | *{len(columns) - max_cols} more columns* |")

                                result.append("")  # Empty line after table

                            # Add examples if detailed level and examples exist
                            if detail_level == "detailed" and table_data.get("examples"):
                                result.append("**Example Queries**:\n")
                                for example in table_data["examples"][:3]:
                                    if isinstance(example, dict):
                                        nl_query = example.get("natural_language", "")
                                        query = example.get("query", "")
                                        if nl_query and query:
                                            result.append(f"*{nl_query}*")
                                            result.append("```q")
                                            result.append(query)
                                            result.append("```\n")

    return "\n".join(result)

# Backward compatibility function
async def generate_schema_description(state):
    """
    Backward compatibility wrapper.
    Routes to enhanced version.
    """
    return await generate_enhanced_schema_description(state)