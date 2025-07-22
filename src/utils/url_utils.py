import re

import requests
import tenacity
from markdownify import markdownify

from common.exceptions import CouldNotReadUrl


def is_arxiv_pdf_or_html_url(url: str) -> bool:
    return bool(re.match(r"https?://arxiv\.org/(pdf|html)/\d+\.\d+(v\d+)?", url))


def convert_to_arxiv_abs_url(url: str) -> str:
    match = re.match(
        r"https?://arxiv\.org/(?:pdf|html|abs)/(\d+\.\d+)(v\d+)?(?:\.pdf)?", url
    )
    if match:
        return f"https://arxiv.org/abs/{match.group(1)}"
    return url


@tenacity.retry(
    wait=tenacity.wait_fixed(5),
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_exception_type(CouldNotReadUrl),
    reraise=True,
)
def get_url_content_as_markdown(url: str) -> str:
    tried_arxiv_fallback = False
    original_url = url

    for _ in range(2):
        try:
            response = requests.get(
                url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}
            )

            if response.status_code == 200:
                content = markdownify(html=response.text)
                return content

        except requests.exceptions.RequestException:
            # If the visit url failed and it is an arxiv html or pdf url fallback on the abs of the article
            if not tried_arxiv_fallback and is_arxiv_pdf_or_html_url(url):
                url = convert_to_arxiv_abs_url(url)
                tried_arxiv_fallback = True
                continue
            break

    raise CouldNotReadUrl(f"Couldn't read the URL: {original_url}")
