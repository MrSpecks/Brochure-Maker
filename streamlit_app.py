import os
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="AI Brochure Maker", page_icon="🧭", layout="wide")


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


def get_openrouter_key() -> str:
    # Primary: Streamlit secrets; Fallback: environment variable; else raise
    key = st.secrets.get("Openrouter_Api_Key") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        # As requested, demonstrate the exact access pattern; do not actually use this value here.
        # Example of direct access if stored exactly in secrets: st.secrets['Openrouter_Api_Key=sk-or-...']
        raise RuntimeError(
            "OpenRouter API key not found. Add Openrouter_Api_Key to .streamlit/secrets.toml"
        )
    return key


@st.cache_data(show_spinner=False)
def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HTTP_HEADERS, timeout=20)
    response.raise_for_status()
    return response.text


@st.cache_data(show_spinner=False)
def extract_relevant_links(seed_url: str, html: str) -> list[str]:
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


def generate_brochure(company_url: str, page_texts: dict[str, str]) -> str:
    # Deferred import to avoid importing if not needed and to keep startup fast
    from openai import OpenAI

    api_key = get_openrouter_key()
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    messages = build_prompt(company_url, page_texts)
    resp = client.chat.completions.create(
        model="openrouter/sonoma-dusk-alpha",
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

# Sidebar inputs
with st.sidebar:
    st.header("Setup")
    st.write(
        "Add `Openrouter_Api_Key` to your `.streamlit/secrets.toml`."
    )
    st.write(
        "The app reads the key from Streamlit secrets."
    )

    default_url = st.session_state.get("input_url", "")
    input_url = st.text_input("Company Website URL", value=default_url, placeholder="https://example.com")
    st.session_state["input_url"] = input_url
    run = st.button("Generate Brochure", type="primary")


col_left, col_right = st.columns([1, 1])

if run:
    if not input_url:
        st.error("Please enter a company website URL.")
    else:
        # Validate URL
        parsed = urlparse(input_url)
        if not parsed.scheme or not parsed.netloc:
            st.error("Please provide a valid URL including http:// or https://.")
        else:
            try:
                with st.spinner("Fetching landing page..."):
                    html = fetch_html(input_url)
                with st.spinner("Finding relevant links (About, Careers, etc.)..."):
                    links = extract_relevant_links(input_url, html)
                with st.spinner("Scraping relevant pages..."):
                    contents = scrape_pages(links)
                with st.spinner("Generating brochure with OpenRouter..."):
                    brochure_md = generate_brochure(input_url, contents)

                with col_left:
                    st.subheader("Status & Raw Data")
                    st.markdown("**Relevant links found:**")
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
                    if brochure_md.strip():
                        st.markdown(brochure_md)
                        st.success("Brochure generated successfully.")
                    else:
                        st.warning("No content returned by the model. Try again or adjust the URL.")

            except requests.exceptions.RequestException as e:
                st.error(f"Network error while fetching pages: {e}")
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")


