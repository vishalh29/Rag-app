# 🚀 Streamlit Cloud Deployment Guide for RAG App

## Setting Up Environment Variables in Streamlit Cloud

When deploying to Streamlit Cloud, you need to configure your API keys through the Streamlit Cloud dashboard instead of using a `.env` file.

### Step-by-Step Instructions:

#### 1. Deploy Your App to Streamlit Cloud
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub account if you haven't already
- Click "New app" and connect to your repository
- Set the main file path to `streamlit_app.py`
- Click "Deploy!"

#### 2. Configure Secrets (Environment Variables)
Once your app is deployed:

1. **Navigate to App Settings:**
   - Go to your deployed app
   - Click the "⚙️" (settings) button in the top-right corner
   - Select "Settings" from the dropdown menu

2. **Add Your API Keys:**
   - In the settings panel, look for the "Secrets" section
   - Click on the "Secrets" tab
   - Add your secrets in TOML format:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
PINECONE_API_KEY = "your-pinecone-api-key-here"
```

3. **Save and Restart:**
   - Click "Save"
   - Your app will automatically restart with the new environment variables

### Getting Your API Keys:

#### OpenAI API Key:
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign in to your account
3. Navigate to "API Keys" in the sidebar
4. Click "Create new secret key"
5. Copy the key and add it to your Streamlit secrets

#### Pinecone API Key:
1. Go to [app.pinecone.io](https://app.pinecone.io)
2. Sign in to your account
3. Go to "API Keys" in the sidebar
4. Copy your API key and add it to your Streamlit secrets

### Important Notes:

- 🔐 **Never commit API keys to your repository** - always use environment variables or secrets
- 🔄 **App restarts automatically** when you update secrets
- 💰 **Monitor usage** - both OpenAI and Pinecone have usage-based pricing
- 🏠 **Local development** still works with `.env` files as before

### Troubleshooting:

#### App Still Shows "API key not found" Error:
1. Check that your secrets are properly formatted in TOML syntax
2. Ensure there are no extra quotes or spaces around your keys
3. Wait 1-2 minutes after saving secrets for the app to restart
4. Check that your API keys are valid and active

#### App is Slow or Timing Out:
1. Pinecone index creation can take 1-2 minutes on first run
2. Large PDF processing may take time - consider smaller files for testing
3. Check Streamlit Cloud resource limits if processing very large documents

### Security Best Practices:

1. **Rotate API keys regularly**
2. **Use different API keys for development and production**
3. **Monitor API usage and set usage limits**
4. **Don't share your deployed app URL if it contains sensitive data**

### Example `.env` File for Local Development:

Create a `.env` file in your project root (for local development only):

```env
OPENAI_API_KEY=your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
```

**Remember:** The `.env` file should be in your `.gitignore` and never committed to version control!
