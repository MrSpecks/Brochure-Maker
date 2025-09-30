import os
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI, APIError

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="AI Brochure Maker", page_icon="🧭", layout="wide")

# Initialize session state for API key and config if not present
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "llm_provider" not in st.session_state:
    st.session_state["llm_provider"] = "OpenRouter"
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = ""

# -----------------------------
# LLM Configuration Mappings
# -----------------------------
# Configuration mapping for providers compatible with the OpenAI Python SDK
LLM_CONFIGS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.2-3b-instruct:free",
        "secret_key_name": "Openrouter_Api_Key",
        "env_key_name": "OPENROUTER_API_KEY",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "secret_key_name": "Openai_Api_Key",
        "env_key_name": "OPENAI_API_KEY",
    },
    # Anthropic and Google are noted as incompatible with the unified OpenAI client below
}


# -----------------------------
# Utilities
# -----------------------------
RELEVANT_KEYWORDS = (
    "about",
    "company",
    "team",
    "leadership",
    "mission",
    "values",
    "careers",
    "jobs",
    "culture",
    "contact",
)

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}


def get_api_key(provider: str) -> str:
    """Retrieves the API key from session state, then secrets, then environment."""
    
    # 1. Check user-input key in session state (highest priority for demo)
    if st.session_state.get("api_key"):
        return st.session_state["api_key"]

    config = LLM_CONFIGS.get(provider, {})
    secret_key = config.get("secret_key_name", "")
    env_key = config.get("env_key_name", "")

    # 2. Check Streamlit secrets
    key = st.secrets.get(secret_key)
    if key:
        return key

    # 3. Check environment variables
    key = os.getenv(env_key)
    if key:
        return key

    # If no key found, raise a user-friendly error
    raise RuntimeError(
        f"API key not found. Please enter your {provider} key in the sidebar."
    )


@st.cache_data(show_spinner=False)
def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HTTP_HEADERS, timeout=20)
    response.raise_for_status()
    return response.text


@st.cache_data(show_spinner=False)
def extract_relevant_links(seed_url: str, html: str) -> list[str]:
    # ... (link extraction logic remains the same) ...
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a")
    candidates: list[str] = []
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href:
            continue
        absolute = urljoin(seed_url, href)
        path = urlparse(absolute).path.lower()
        if any(keyword in path for keyword in RELEVANT_KEYWORDS):
            candidates.append(absolute)
    # de-duplicate while preserving order
    seen = set()
    unique_links: list[str] = []
    for link in candidates:
        if link not in seen:
            unique_links.append(link)
            seen.add(link)
    return unique_links[:15]


@st.cache_data(show_spinner=False)
def scrape_pages(urls: list[str]) -> dict[str, str]:
    # ... (page scraping logic remains the same) ...
    contents: dict[str, str] = {}
    for u in urls:
        try:
            html = fetch_html(u)
            soup = BeautifulSoup(html, "html.parser")
            body = soup.body
            if body is None:
                contents[u] = ""
                continue
            for irrelevant in body(["script", "style", "img", "input", "svg"]):
                irrelevant.decompose()
            text = body.get_text(separator="\n", strip=True)
            contents[u] = text
        except Exception:
            contents[u] = ""
    return contents


def build_prompt(company_url: str, page_texts: dict[str, str]) -> list[dict[str, str]]:
    # ... (prompt building logic remains the same) ...
    system_prompt = (
        "You are an assistant that analyzes content from a company's website and "
        "creates a concise, professional brochure in Markdown for prospective customers, "
        "investors, and recruits. Include sections like Overview, What We Do, Customers, "
        "Products/Services, Culture & Values, Careers, and Contact. Maintain a clear, "
        "approachable tone and avoid marketing fluff."
    )
    joined = []
    for url, txt in page_texts.items():
        if not txt:
            continue
        joined.append(f"URL: {url}\n---\n{txt}")
    user_prompt = (
        f"Company website: {company_url}\n\n"
        "From the following scraped content, draft a single, well-structured brochure in Markdown. "
        "Be selective and summarize. Avoid repeating navigation or boilerplate.\n\n"
        + "\n\n".join(joined[:6])
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# --- New Cached Function to Fetch Models ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_available_models(provider: str, api_key: str) -> list[str]:
    """Fetches the list of available models for a given provider/key."""
    if provider not in LLM_CONFIGS:
        return []

    config = LLM_CONFIGS[provider]
    base_url = config['base_url']
    
    try:
        # Use the OpenAI client to query the /models endpoint
        client = OpenAI(api_key=api_key, base_url=base_url)
        models_response = client.models.list()
        
        # OpenRouter returns models directly. OpenAI returns models with 'id'
        model_names = [m.id for m in models_response.data]
        
        # Simple filter: only include models that support chat completions (optional, but good)
        # For simplicity, we'll just return all names
        return sorted(model_names)
        
    except APIError as e:
        # Catch authentication/rate limit errors and inform the user
        st.warning(f"Could not fetch models for {provider}. Check API key validity. Error: {e.status_code}")
        return []
    except Exception as e:
        st.warning(f"An unexpected error occurred while fetching models: {e}")
        return []


# --- Modified generate_brochure to accept model_name ---
def generate_brochure(company_url: str, page_texts: dict[str, str], provider: str, model_name: str) -> str:
    
    api_key = get_api_key(provider)
    messages = build_prompt(company_url, page_texts)

    config = LLM_CONFIGS.get(provider)
    if not config:
        return f"Error: Unsupported provider '{provider}'. Could not configure API client."
        
    base_url = config['base_url']
    
    # NOTE: Anthropic and Google APIs require their specific Python client libraries.
    # For a unified code structure, we only support services compatible with the 
    # OpenAI-like API structure (OpenAI, OpenRouter).
    if provider not in LLM_CONFIGS:
         return f"Provider '{provider}' selected. This demo currently only supports OpenAI-compatible APIs (OpenAI, OpenRouter, etc.) for a unified code structure."

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    resp = client.chat.completions.create(
        model=model_name, # Use the dynamically selected model
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


# -----------------------------
# UI
# -----------------------------
st.title("AI Brochure Maker")
st.caption(
    "Automatically scrape a company website, select relevant pages, and generate a polished brochure draft."
)


# --- API Key Configuration Component ---
def llm_config_selector():
    st.subheader("API Key & LLM Setup")
    
    # 1. Provider Selection
    provider = st.selectbox(
        "Select LLM Provider:",
        options=["OpenRouter", "OpenAI", "Anthropic", "Google"],
        key="llm_provider",
        help="Select the API provider you want to use."
    )
    
    # 2. Key Input (Stored in session state)
    st.text_input(
        f"Paste your {provider} API Key here:",
        type="password",
        key="api_key",
        help=f"Your key is only stored in the current browser session and is never saved.",
        placeholder="sk-..."
    )

    # 3. Key Status Check and Model Loading
    model_options = []
    
    if provider in LLM_CONFIGS:
        try:
            current_key = get_api_key(provider)
            key_status = "✅ Key Loaded"
            
            # Fetch models dynamically only if key is present
            if current_key:
                with st.spinner("Fetching models..."):
                    model_options = fetch_available_models(provider, current_key)
            
            if not model_options:
                st.warning("Could not fetch models. Check your API key or network connection.")
                default_model = LLM_CONFIGS[provider]['default_model']
                model_options = [default_model]
                
        except RuntimeError:
            key_status = "❌ Key Missing"
            default_model = LLM_CONFIGS[provider]['default_model'] if provider in LLM_CONFIGS else "No Model Available"
            model_options = [default_model]
            
        st.info(f"Key Status: **{key_status}**")
        
        # 4. Model Selection Box
        st.selectbox(
            "Select Model:",
            options=model_options,
            key="selected_model",
            help="The list is dynamically loaded from the API. Only Chat models are recommended."
        )

    else:
        # For non-compatible providers like Anthropic/Google
        st.warning(f"Note: {provider} requires a different client. Try OpenAI or OpenRouter for this demo.")
        st.session_state["selected_model"] = "N/A" # Clear model state for incompatible providers


# Sidebar inputs
with st.sidebar:
    llm_config_selector() # Call the configuration function
    st.markdown("---")
    st.subheader("Website Input")
    
    default_url = st.session_state.get("input_url", "")
    input_url = st.text_input("Company Website URL", value=default_url, placeholder="https://example.com")
    st.session_state["input_url"] = input_url
    run = st.button("Generate Brochure", type="primary")


col_left, col_right = st.columns([1, 1])

if run:
    if not input_url:
        st.error("Please enter a company website URL.")
    elif st.session_state["llm_provider"] not in LLM_CONFIGS:
        st.error(f"The selected provider ({st.session_state['llm_provider']}) is not supported for LLM generation in this version.")
    else:
        # Validate URL and proceed
        parsed = urlparse(input_url)
        if not parsed.scheme or not parsed.netloc:
            st.error("Please provide a valid URL including http:// or https://.")
        else:
            try:
                # Check for API key before starting long processes
                provider = st.session_state["llm_provider"]
                model_name = st.session_state["selected_model"]
                get_api_key(provider) # This will raise RuntimeError if key is missing

                # --- Core Process ---
                with st.spinner("Fetching landing page..."):
                    html = fetch_html(input_url)
                with st.spinner("Finding relevant links (About, Careers, etc.)..."):
                    links = extract_relevant_links(input_url, html)
                with st.spinner("Scraping relevant pages..."):
                    contents = scrape_pages(links)
                
                # Use the selected provider and model in the generation step
                with st.spinner(f"Generating brochure with {provider}/{model_name}..."):
                    brochure_md = generate_brochure(input_url, contents, provider, model_name)
                # --- End Core Process ---

                # Display Results
                with col_left:
                    st.subheader("Status & Raw Data")
                    st.markdown("**Relevant links found:**")
                    # ... (display links logic) ...
                    if links:
                        for l in links:
                            st.write(f"- {l}")
                    else:
                        st.write("No relevant links discovered.")

                    for l in links:
                        txt = contents.get(l, "")
                        with st.expander(l, expanded=False):
                            if txt:
                                st.text(txt[:10000])
                            else:
                                st.info("No readable content extracted.")

                with col_right:
                    st.subheader("Final Brochure Draft")
                    if "Error:" in brochure_md: # Check for the error message from generate_brochure
                         st.error(brochure_md)
                    elif brochure_md.strip():
                        st.markdown(brochure_md)
                        st.success("Brochure generated successfully using **" + model_name + "**.")
                    else:
                        st.warning("No content returned by the model. Try again or adjust the URL.")

            except requests.exceptions.RequestException as e:
                st.error(f"Network error while fetching pages: {e}")
            except RuntimeError as e:
                # Catches the RuntimeError from get_api_key
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")