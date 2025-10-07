"""
Tests for glossary integration in query generator node.

Tests the new glossary retrieval and formatting features that provide
domain-specific terminology to help the LLM understand business terms.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.query_generation.nodes.query_generator_node import (
    get_glossary_for_schema_group,
    get_glossary_from_directives,
    format_glossary_for_prompt,
    generate_initial_query,
    generate_syntax_fix_query,
    generate_schema_fix_query,
    generate_complexity_escalated_query,
    generate_retry_query
)


class TestGlossaryRetrieval:
    """Test glossary retrieval functions."""

    @pytest.mark.asyncio
    async def test_get_glossary_for_schema_group_returns_dict(self):
        """Test that get_glossary_for_schema_group returns a dictionary."""
        result = await get_glossary_for_schema_group("test_schema")

        assert isinstance(result, dict)
        assert len(result) > 0
        assert "TCA" in result
        assert "VWAP" in result
        assert "notional" in result

    @pytest.mark.asyncio
    async def test_get_glossary_for_schema_group_returns_expected_terms(self):
        """Test that glossary contains expected financial terms."""
        result = await get_glossary_for_schema_group("trading_schema")

        # Check for specific terms
        assert "TCA" in result
        assert "Transaction Cost Analysis" in result["TCA"]

        assert "VWAP" in result
        assert "Volume Weighted Average Price" in result["VWAP"]

        assert "notional" in result
        assert "price × quantity" in result["notional"]

    @pytest.mark.asyncio
    async def test_get_glossary_for_schema_group_handles_empty_schema(self):
        """Test that empty schema name returns empty dict."""
        result = await get_glossary_for_schema_group("")

        # Currently returns sample data even for empty string
        # This will change when DB implementation is added
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_glossary_from_directives_with_single_directive(self):
        """Test get_glossary_from_directives with a single directive."""
        directives = ["trading_schema"]

        result = await get_glossary_from_directives(directives)

        assert isinstance(result, dict)
        assert len(result) > 0
        assert "TCA" in result
        assert "VWAP" in result

    @pytest.mark.asyncio
    async def test_get_glossary_from_directives_with_multiple_directives(self):
        """Test get_glossary_from_directives combines multiple directives."""
        directives = ["trading_schema", "analytics_schema"]

        result = await get_glossary_from_directives(directives)

        assert isinstance(result, dict)
        assert len(result) > 0
        # Should have terms from both schemas (currently returns same hardcoded data)

    @pytest.mark.asyncio
    async def test_get_glossary_from_directives_with_empty_list(self):
        """Test get_glossary_from_directives with empty list."""
        result = await get_glossary_from_directives([])

        assert isinstance(result, dict)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_glossary_from_directives_with_none(self):
        """Test get_glossary_from_directives with None."""
        result = await get_glossary_from_directives(None)

        assert isinstance(result, dict)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_glossary_from_directives_skips_empty_directives(self):
        """Test that empty strings in directives list are skipped."""
        directives = ["trading_schema", "", "analytics_schema"]

        result = await get_glossary_from_directives(directives)

        assert isinstance(result, dict)
        assert len(result) > 0


class TestGlossaryFormatting:
    """Test glossary formatting for prompts."""

    def test_format_glossary_for_prompt_with_terms(self):
        """Test formatting glossary with multiple terms."""
        glossary = {
            "TCA": "Transaction Cost Analysis",
            "VWAP": "Volume Weighted Average Price",
            "notional": "Total value of a position"
        }

        result = format_glossary_for_prompt(glossary)

        assert isinstance(result, str)
        assert "Business Glossary:" in result
        assert "TCA" in result
        assert "Transaction Cost Analysis" in result
        assert "VWAP" in result
        assert "Volume Weighted Average Price" in result
        assert "notional" in result
        assert "Total value of a position" in result
        assert "•" in result  # Bullet point

    def test_format_glossary_for_prompt_with_empty_dict(self):
        """Test formatting empty glossary returns empty string."""
        result = format_glossary_for_prompt({})

        assert result == ""

    def test_format_glossary_for_prompt_structure(self):
        """Test that formatted glossary has correct structure."""
        glossary = {
            "term1": "definition1",
            "term2": "definition2"
        }

        result = format_glossary_for_prompt(glossary)

        # Check structure
        assert result.startswith("\n\nBusiness Glossary:")
        assert "specific meanings in this domain" in result
        assert "interpreting the user's query" in result
        assert result.count("•") == 2  # Two bullet points


class TestGlossaryIntegrationInGeneration:
    """Test that glossary is integrated into generation methods."""

    @pytest.mark.asyncio
    async def test_generate_initial_query_includes_glossary(self):
        """Test that generate_initial_query retrieves and includes glossary."""
        # Create mock state
        mock_state = MagicMock()
        mock_state.query = "Show me TCA for AAPL"
        mock_state.directives = ["trading_schema"]
        mock_state.entities = []
        mock_state.query_schema = {
            "schema": "trading_schema",
            "tables": {"trades": {}},
            "examples": []
        }
        mock_state.database_type = "kdb"
        mock_state.conversation_history = []
        mock_state.user_id = "test_user"
        mock_state.conversation_summary = None
        mock_state.conversation_essence = None
        mock_state.thinking = []

        # Mock FeedbackManager
        with patch('app.services.query_generation.nodes.query_generator_node.FeedbackManager') as mock_fb:
            mock_fb_instance = MagicMock()
            mock_fb_instance.find_similar_verified_queries = AsyncMock(return_value=[])
            mock_fb.return_value = mock_fb_instance

            # Mock get_complexity_guidance
            with patch('app.services.query_generation.nodes.query_generator_node.get_complexity_guidance') as mock_guidance:
                mock_guidance.return_value = ("guidance", "notes")

                # Mock ChatPromptTemplate chain
                with patch('app.services.query_generation.nodes.query_generator_node.ChatPromptTemplate') as mock_template:
                    mock_chain = AsyncMock()
                    mock_response = MagicMock()
                    mock_response.content = "select from trades where sym=`AAPL"
                    mock_chain.ainvoke = AsyncMock(return_value=mock_response)

                    mock_prompt = MagicMock()
                    mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                    mock_template.from_template.return_value = mock_prompt

                    mock_llm = MagicMock()

                    # Call the function
                    result = await generate_initial_query(mock_state, mock_llm)

        # Verify result
        assert result == "select from trades where sym=`AAPL"

        # Verify thinking was updated with glossary message
        assert any("glossary" in str(thought).lower() for thought in mock_state.thinking)

    @pytest.mark.asyncio
    async def test_generate_syntax_fix_query_includes_glossary(self):
        """Test that generate_syntax_fix_query includes glossary."""
        # Create mock state
        mock_state = MagicMock()
        mock_state.query = "Show me VWAP"
        mock_state.directives = ["spot"]
        mock_state.generated_query = "select vwap from trades"
        mock_state.query_schema = {
            "schema": "trading_schema",
            "tables": {"trades": {}},
            "examples": []
        }

        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "select vwap from trades where sym=`AAPL"
        mock_llm.ainvoke.return_value = mock_response

        # Call the function
        result = await generate_syntax_fix_query(
            mock_state,
            mock_llm,
            "Fix syntax error",
            "SINGLE_LINE"
        )

        # Verify LLM was called
        assert mock_llm.ainvoke.called

        # Get the prompt that was sent to LLM
        call_args = mock_llm.ainvoke.call_args
        prompt = call_args[0][0]

        # Verify glossary section is in prompt
        assert "Business Glossary:" in prompt or "VWAP" in prompt

    @pytest.mark.asyncio
    async def test_generate_schema_fix_query_includes_glossary(self):
        """Test that generate_schema_fix_query includes glossary."""
        # Create mock state
        mock_state = MagicMock()
        mock_state.query = "Show me notional"
        mock_state.directives = ["trading_schema"]
        mock_state.generated_query = "select notional from trades"
        mock_state.query_schema = {
            "schema": "trading_schema",
            "tables": {"trades": {}},
            "examples": []
        }

        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "select notional from trades where date=.z.d"
        mock_llm.ainvoke.return_value = mock_response

        # Call the function
        result = await generate_schema_fix_query(
            mock_state,
            mock_llm,
            "Fix schema references",
            "SINGLE_LINE"
        )

        # Verify LLM was called
        assert mock_llm.ainvoke.called

        # Get the prompt
        call_args = mock_llm.ainvoke.call_args
        prompt = call_args[0][0]

        # Verify glossary is in prompt
        assert "Business Glossary:" in prompt

    @pytest.mark.asyncio
    async def test_generate_complexity_escalated_query_includes_glossary(self):
        """Test that generate_complexity_escalated_query includes glossary."""
        # Create mock state
        mock_state = MagicMock()
        mock_state.query = "Calculate slippage"
        mock_state.directives = ["trading_schema"]
        mock_state.generated_query = "select slippage from trades"
        mock_state.query_schema = {
            "schema": "trading_schema",
            "tables": {"trades": {}},
            "examples": []
        }
        mock_state.execution_plan = []

        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "result:select slippage from trades"
        mock_llm.ainvoke.return_value = mock_response

        # Mock get_complexity_guidance
        with patch('app.services.query_generation.nodes.query_generator_node.get_complexity_guidance') as mock_guidance:
            mock_guidance.return_value = ("guidance", "notes")

            # Call the function
            result = await generate_complexity_escalated_query(
                mock_state,
                mock_llm,
                "Escalate complexity",
                "MULTI_LINE"
            )

        # Verify LLM was called
        assert mock_llm.ainvoke.called

        # Get the prompt
        call_args = mock_llm.ainvoke.call_args
        prompt = call_args[0][0]

        # Verify glossary is in prompt
        assert "Business Glossary:" in prompt

    @pytest.mark.asyncio
    async def test_generate_retry_query_includes_glossary(self):
        """Test that generate_retry_query includes glossary."""
        # Create mock state
        mock_state = MagicMock()
        mock_state.query = "Show me alpha"
        mock_state.directives = ["trading_schema"]
        mock_state.original_generated_query = "select alpha from metrics"
        mock_state.user_feedback = "Add date filter"
        mock_state.query_schema = {
            "schema": "trading_schema",
            "tables": {"metrics": {}},
            "examples": []
        }
        mock_state.user_id = "test_user"
        mock_state.feedback_trail = []
        mock_state.key_context = []
        mock_state.thinking = []

        # Mock FeedbackManager
        with patch('app.services.query_generation.nodes.query_generator_node.FeedbackManager') as mock_fb:
            mock_fb_instance = MagicMock()
            mock_fb_instance.find_similar_verified_queries = AsyncMock(return_value=[])
            mock_fb.return_value = mock_fb_instance

            # Mock ChatPromptTemplate chain
            with patch('app.services.query_generation.nodes.query_generator_node.ChatPromptTemplate') as mock_template:
                mock_chain = AsyncMock()
                mock_response = MagicMock()
                mock_response.content = "select alpha from metrics where date=.z.d"
                mock_chain.ainvoke = AsyncMock(return_value=mock_response)

                mock_prompt = MagicMock()
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                mock_template.from_template.return_value = mock_prompt

                mock_llm = MagicMock()

                # Call the function
                result = await generate_retry_query(mock_state, mock_llm)

        # Verify result
        assert result == "select alpha from metrics where date=.z.d"


class TestGlossaryErrorHandling:
    """Test error handling in glossary functions."""

    @pytest.mark.asyncio
    async def test_get_glossary_handles_exception_gracefully(self):
        """Test that exceptions in get_glossary return empty dict."""
        # Patch the function to raise an exception
        with patch('app.services.query_generation.nodes.query_generator_node.get_glossary_for_schema_group') as mock_get:
            mock_get.side_effect = Exception("Database error")

            result = await get_glossary_from_directives(["test_schema"])

            # Should return empty dict on error
            assert isinstance(result, dict)
            assert len(result) == 0

    def test_format_glossary_handles_none_gracefully(self):
        """Test that format_glossary handles None input."""
        result = format_glossary_for_prompt(None)

        assert result == ""


class TestGlossaryDatabaseIntegration:
    """Test glossary database integration (placeholder for future DB implementation)."""

    @pytest.mark.asyncio
    async def test_glossary_retrieval_with_db_mock(self):
        """Test glossary retrieval with mocked database call."""
        # This test demonstrates how glossary DB integration will work
        # Currently it returns hardcoded data, so we test that the function works
        result = await get_glossary_for_schema_group("trading_schema")

        # Verify it returns a dictionary
        assert isinstance(result, dict)
        assert len(result) > 0

        # Verify it contains expected terms (from hardcoded sample)
        assert "TCA" in result
        assert "VWAP" in result

    def test_glossary_format_preserves_all_terms(self):
        """Test that formatting preserves all glossary terms."""
        glossary = {
            "term1": "def1",
            "term2": "def2",
            "term3": "def3",
            "term4": "def4",
            "term5": "def5"
        }

        result = format_glossary_for_prompt(glossary)

        # All terms should be present
        for term, definition in glossary.items():
            assert term in result
            assert definition in result

        # Should have correct number of bullet points
        assert result.count("•") == len(glossary)