# Troubleshooting Guide

This guide helps diagnose and resolve common issues with the Query Generator application.

## Database Issues

### PostgreSQL Connection Problems

**Symptoms:**
- Error: "Connection refused"
- Error: "Could not connect to server"
- Error: "Role 'user' does not exist"

**Solutions:**

1. **Verify PostgreSQL is running:**
   ```bash
   sudo systemctl status postgresql
   # or
   ps aux | grep postgres
   ```

2. **Check database connection string:**
   ```
   # In .env file
   DATABASE_URL=postgresql://username:password@localhost:5432/query_generator
   ```
   Ensure the username, password, host, port, and database name are correct.

3. **Verify database exists:**
   ```bash
   psql -U postgres -c "SELECT datname FROM pg_database WHERE datname='query_generator';"
   ```
   If it doesn't exist, create it:
   ```bash
   psql -U postgres -c "CREATE DATABASE query_generator;"
   ```

4. **Check PostgreSQL authentication settings:**
   Edit pg_hba.conf to allow password authentication:
   ```
   # IPv4 local connections:
   host    all             all             127.0.0.1/32            md5
   ```

### pgvector Extension Issues

**Symptoms:**
- Error: "Extension 'vector' does not exist"
- Error: "Function 'vector_cosine_ops' does not exist"
- Vector search doesn't work

**Solutions:**

1. **Verify pgvector is installed:**
   ```bash
   psql -U postgres -d query_generator -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
   ```

2. **Install pgvector:**
   ```bash
   # Install from source
   git clone --branch v0.4.4 https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   make install
   
   # Then in your database:
   psql -U postgres -d query_generator -c "CREATE EXTENSION vector;"
   ```

3. **Check vector dimensions:**
   If you see errors about vector dimensions, ensure your embedding model's dimensions match the database schema:
   ```bash
   # Check column definition
   psql -U postgres -d query_generator -c "\d table_definitions"
   ```
   The `embedding vector(768)` should match your embedding model dimensions.

### Database Schema Initialization Issues

**Symptoms:**
- Error: "Relation does not exist"
- Missing tables after initialization
- Schema upload fails

**Solutions:**

1. **Run the initialization script:**
   ```bash
   python scripts/init_database.py
   ```

2. **Check for SQL errors:**
   Look for error messages during initialization. Common issues:
   - Syntax errors in SQL scripts
   - Missing permissions
   - Tables already exist

3. **Verify tables were created:**
   ```bash
   psql -U postgres -d query_generator -c "\dt"
   ```

4. **Reset the database if needed:**
   ```bash
   psql -U postgres -c "DROP DATABASE query_generator;"
   psql -U postgres -c "CREATE DATABASE query_generator;"
   python scripts/init_database.py
   ```

## LLM Connection Issues

### Google Vertex AI Connection Failures

**Symptoms:**
- Error: "Failed to authenticate with Google Cloud services"
- Error: "Permission denied for project [project-id]"
- Error: "Model not found: gemini-1.5-pro-002"

**Solutions:**

1. **Check Google Cloud credentials file:**
   ```bash
   cat google-credentials.json
   ```
   Ensure the file is properly formatted JSON and not corrupted.

2. **Verify environment variables:**
   ```bash
   echo $GOOGLE_PROJECT_ID
   echo $GOOGLE_CREDENTIALS_PATH
   ```
   Make sure these are set correctly.

3. **Check service account permissions:**
   - Go to Google Cloud Console > IAM & Admin > IAM
   - Ensure your service account has at least these roles:
     - Vertex AI User
     - Vertex AI Service Agent

4. **Check the model name:**
   - Verify that the model name in your settings (default: "gemini-1.5-pro-002") is correct
   - Try a different model version if available

5. **Run the test script:**
   ```bash
   python test_llm_connection.py
   ```
   Check the detailed error messages.

### Claude API Connection Failures

**Symptoms:**
- Error: "Invalid API key provided"
- Error: "Rate limit exceeded"
- Error: "Model not available: claude-3-sonnet-20240229"

**Solutions:**

1. **Check API key:**
   - Verify you're using the correct API key
   - Make sure it hasn't expired
   - Check for special characters or spaces that might have been introduced erroneously

2. **Check usage limits:**
   - Verify you haven't exceeded your quota
   - Consider implementing rate limiting if you're hitting limits

3. **Check model availability:**
   - Verify the model name is correct
   - Make sure your account has access to the requested model

4. **Test with cURL:**
   ```bash
   curl -X POST \
     -H "x-api-key: $API_KEY_ANTHROPIC" \
     -H "content-type: application/json" \
     -d '{
       "model": "claude-3-sonnet-20240229",
       "max_tokens": 1000,
       "messages": [{"role": "user", "content": "Hello"}]
     }' \
     https://api.anthropic.com/v1/messages
   ```

## Embedding Service Issues

**Symptoms:**
- Error: "Failed to generate embedding"
- Vector search returns no results
- Schema upload fails during embedding

**Solutions:**

1. **Check embedding model settings:**
   ```
   # In .env file
   GOOGLE_EMBEDDING_MODEL_NAME=text-embedding-005
   ```

2. **Verify Google Cloud permissions:**
   - Ensure your service account has access to the embedding model
   - Test the embedding API directly:
   
   ```python
   from google.cloud import aiplatform
   
   # Initialize Vertex AI
   aiplatform.init(project="your-project", location="us-central1")
   
   # Create the endpoint
   endpoint = aiplatform.VertexEndpoint(
       endpoint_name="textembedding-gecko"
   )
   
   # Get embeddings
   response = endpoint.predict(
       instances=[{"content": "Test embedding"}]
   )
   print(response)
   ```

3. **Check vector dimensions:**
   - The database schema creates vector columns with dimension 768
   - If your embedding model has a different dimension, update the SQL schema

4. **Debug embedding generation:**
   - Add detailed logging to `embedding_provider.py`
   - Try a simpler text string for embedding

## Schema Management Issues

### Schema Upload Failures

**Symptoms:**
- Error when uploading schema files
- Schema uploads but tables aren't found
- Vector search doesn't return expected results

**Solutions:**

1. **Validate schema JSON format:**
   ```bash
   python -m json.tool your_schema.json
   ```
   This will format the JSON and validate its syntax.

2. **Check schema structure:**
   Ensure your schema follows the expected structure:
   - Has a top-level "group" field
   - Contains "schemas" array
   - Each schema has "schema" array with tables

3. **Inspect schema with jq:**
   ```bash
   jq '.group' your_schema.json
   jq '.schemas[0].schema[0].tables | length' your_schema.json
   ```

4. **Try with a simpler schema:**
   Create a minimal schema file with a single table and fewer columns.

5. **Check database after upload:**
   ```bash
   psql -U postgres -d query_generator -c "SELECT * FROM schema_definitions;"
   psql -U postgres -d query_generator -c "SELECT * FROM table_definitions;"
   ```

### Vector Search Issues

**Symptoms:**
- No schemas found for a query
- Irrelevant schemas returned
- Low similarity scores

**Solutions:**

1. **Test vector search directly:**
   ```bash
   python scripts/test_vector_search.py "Show me AAPL trades from today"
   ```

2. **Adjust similarity threshold:**
   ```
   # In .env file
   SCHEMA_SIMILARITY_THRESHOLD=0.5  # Lower for more results
   ```

3. **Check embedding quality:**
   - Ensure table descriptions are clear and relevant
   - Add more domain-specific terminology to descriptions
   - Improve column descriptions

4. **Verify tables were embedded:**
   ```bash
   psql -U postgres -d query_generator -c "SELECT id, name, embedding IS NOT NULL FROM table_definitions;"
   ```

5. **Rebuild embeddings if needed:**
   You may need to update the schema management service to re-embed tables.

## Query Generation Issues

### Query Not Generating

**Symptoms:**
- Empty query result
- Error message in response
- Request times out

**Solutions:**

1. **Check logs:**
   - Look at server logs for error messages
   - If in debug mode, detailed error information should be available

2. **Verify schema retrieval:**
   - Check if schemas are being found via vector search
   - Adjust similarity threshold if no schemas are found
   - Test with the vector search script

3. **Review the thinking steps:**
   - Examine the `thinking` array in the response
   - Look for errors or unexpected behavior in the pipeline
   - Identify which step is failing

4. **Try a simple query:**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me all trades"}'
   ```

5. **Check LLM connectivity:**
   - Run the LLM test script
   - Verify API keys and permissions

### Incorrect Query Syntax

**Symptoms:**
- Query has syntax errors
- Query uses SQL syntax instead of KDB/Q syntax
- Query doesn't match user intent

**Solutions:**

1. **Check prompt templates:**
   - Review `app/services/query_generation/prompts/generator_prompts.py`
   - Ensure KDB/Q syntax notes are clear and accurate

2. **Add more examples:**
   - Enhance schema files with more example query pairs
   - Focus on examples similar to problematic queries

3. **Improve schema descriptions:**
   - Make table and column descriptions more detailed
   - Include specific formatting requirements

4. **Adjust model parameters:**
   - Try lower temperature values (e.g., 0.1) for more deterministic results
   - Temperature controls randomness; lower values are more conservative

5. **Use the retry endpoint:**
   - Send feedback via the `/retry` endpoint
   - This triggers a refined query generation with feedback

### Pipeline Node Failures

**Symptoms:**
- Error in a specific pipeline node
- Unexpected state between nodes
- Missing or invalid data in state

**Solutions:**

1. **Enable DEBUG mode:**
   ```
   DEBUG=True
   ```

2. **Add more logging:**
   Add detailed logging to each node:
   ```python
   logger.debug("State before processing: %s", state)
   # Process state
   logger.debug("State after processing: %s", state)
   ```

3. **Isolate the failing node:**
   - Check the `thinking` output to identify the failing node
   - Test the node function directly

4. **Check state transitions:**
   - Verify that each node is updating the state correctly
   - Ensure required fields are present in the state

5. **Adjust refinement settings:**
   - Increase max refinements to allow more attempts:
   ```python
   max_refinements: int = Field(default=3, description="Maximum number of refinement attempts")
   ```

## WebSocket Issues

### Connection Problems

**Symptoms:**
- WebSocket connection fails to establish
- Connection drops shortly after connecting
- Error: "Connection closed abnormally"

**Solutions:**

1. **Check CORS settings:**
   - Verify your client domain is in the `CORS_ORIGINS` list
   - Check browser console for CORS errors

2. **Check server logs:**
   - Look for WebSocket-related error messages
   - Pay attention to connection initialization errors

3. **Test with a simple client:**
   ```javascript
   const socket = io("ws://localhost:8000/ws");
   socket.on("connect", () => console.log("Connected"));
   socket.on("disconnect", () => console.log("Disconnected"));
   socket.on("error", (err) => console.error("Error:", err));
   ```

4. **Verify Socket.IO version compatibility:**
   - Make sure client and server versions are compatible
   - Check for version-specific issues

5. **Adjust WebSocket timeouts:**
   ```
   # In .env file
   WS_PING_INTERVAL=30
   WS_PING_TIMEOUT=60
   ```

### Message Handling Issues

**Symptoms:**
- Messages not being received
- Server not responding to events
- Invalid message format errors

**Solutions:**

1. **Verify event names:**
   - Ensure client and server are using the same event names
   - Check for typos in event names

2. **Validate message format:**
   - Ensure messages follow the expected format
   - Check that JSON data is properly formatted

3. **Debug with event listeners:**
   ```javascript
   // Add listeners for all events
   socket.onAny((event, ...args) => {
     console.log("Event:", event, "Args:", args);
   });
   ```

4. **Test with minimal messages:**
   Start with the simplest possible messages and gradually add complexity.

5. **Enable Socket.IO debugging:**
   ```javascript
   // Client-side
   const socket = io("ws://localhost:8000/ws", { debug: true });
   ```

## Performance Issues

### Slow Query Generation

**Symptoms:**
- Query generation takes a long time
- Timeouts during generation
- High CPU usage

**Solutions:**

1. **Profile the application:**
   Use a tool like `cProfile` to identify bottlenecks:
   ```bash
   python -m cProfile -o profile.out run.py
   ```

2. **Check LLM response times:**
   - Add timing logs around LLM calls
   - Consider switching to a faster model for some operations

3. **Optimize vector search:**
   - Limit the number of tables returned
   - Add indices to the database
   - Consider caching common queries

4. **Adjust LangGraph execution:**
   - Simplify the pipeline if possible
   - Reduce the number of LLM calls

5. **Scale horizontally:**
   - Deploy multiple instances behind a load balancer
   - Consider asynchronous processing for long-running operations

### Memory Issues

**Symptoms:**
- Memory usage grows over time
- Application crashes with out-of-memory errors
- Slow performance after running for a while

**Solutions:**

1. **Monitor memory usage:**
   ```bash
   ps -o pid,user,%mem,command ax | grep python
   ```

2. **Check for memory leaks:**
   - Look for objects that aren't being garbage collected
   - Check for large objects being stored in memory

3. **Limit context size:**
   - Reduce the size of conversation history kept in memory
   - Limit the amount of schema data loaded

4. **Implement memory limits:**
   - Set maximum limits for array sizes
   - Implement pagination for large results

5. **Consider database storage:**
   - Store large data in the database instead of memory
   - Use streaming responses for large results

## Deployment Issues

### Docker Deployment Issues

**Symptoms:**
- Container fails to start
- Container crashes after starting
- Container runs but application is inaccessible

**Solutions:**

1. **Check Docker logs:**
   ```bash
   docker logs query-generator
   ```

2. **Verify environment variables:**
   - Ensure all required environment variables are passed to the container
   - Check for missing or invalid values

3. **Check Docker network:**
   - Verify port mappings
   - Check network connectivity between containers if using Docker Compose

4. **Inspect running container:**
   ```bash
   docker exec -it query-generator bash
   ```
   Then check application status, logs, etc.

5. **Rebuild with no cache:**
   ```bash
   docker build --no-cache -t query-generator .
   ```

### Cloud Deployment Issues

**Symptoms:**
- Deployment fails
- Application starts but is unavailable
- Errors in cloud logs

**Solutions:**

1. **Check cloud logs:**
   - GCP: Check Cloud Run or App Engine logs
   - AWS: Check CloudWatch logs
   - Azure: Check App Service logs

2. **Verify cloud permissions:**
   - Ensure the service has access to required APIs
   - Check IAM roles and permissions

3. **Test locally with same configuration:**
   - Use the same environment variables
   - Use the same database settings

4. **Check for cloud-specific issues:**
   - Resource limits
   - Network constraints
   - API quotas

5. **Deploy with debug mode:**
   Enable debug mode temporarily to get more detailed logs.

## Common Error Messages

### "Failed to generate embedding"

**Cause**: Issues with the embedding service, credentials, or network.

**Solution**:
1. Verify Google Cloud credentials
2. Check permissions for the embedding API
3. Ensure the embedding model exists and is accessible
4. Try a different model or region

### "No schema found for query"

**Cause**: Vector search didn't find any relevant schemas, or schemas aren't in the database.

**Solution**:
1. Upload schemas with the import script
2. Lower the similarity threshold
3. Improve table descriptions
4. Test vector search directly

### "Cannot connect to database"

**Cause**: PostgreSQL is not running, incorrect credentials, or network issues.

**Solution**:
1. Check PostgreSQL is running
2. Verify credentials in `.env`
3. Test connection with `psql`
4. Check database exists and has correct schema

### "LLM response error"

**Cause**: Issues with the LLM API, credentials, quotas, or network.

**Solution**:
1. Check API keys
2. Verify model names
3. Check for rate limiting or quota issues
4. Test LLM connections directly

### "Invalid schema format"

**Cause**: Schema JSON doesn't match the expected format.

**Solution**:
1. Validate JSON syntax
2. Check schema structure against documentation
3. Start with a simple schema
4. Compare with working examples
