import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from text_split import split_text
import json
import time

def fetch_icml_papers():
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", device_map='cuda')
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = "en_XX" # 设置源语言为英语

    # ICML 会议主页 URL
    # url = 'https://proceedings.mlr.press/v202/'

    # CoRL
    url = 'https://proceedings.mlr.press/v229/'

    # 发送 HTTP 请求获取页面内容
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功

    # 使用 BeautifulSoup 解析页面内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有文章信息
    papers = soup.find_all('div', class_='paper')

    paper_lists = []

    # 遍历每篇文章，提取信息
    for paper in tqdm(papers, desc="Fetching ICML Papers"):
        title = paper.find('p', class_='title').text.strip()
        # 标题翻译
        encoded_title = tokenizer(title, return_tensors="pt").to('cuda')
        generated_tokens = model.generate(
            **encoded_title,
            forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
        )
        ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        title_chinese = ans[0]
        # authors = paper.find('div', class_='maincardFooter').text.strip()
        authors = paper.find('span', class_='authors').text.strip().replace('\xa0', ' ')
        links = paper.find('p', class_='links')

        links = links.find_all('a')
        
        paper_url = links[0]['href']
        paper_pdf = links[1]['href']

        response_paper = requests.get(paper_url)
        response_paper.raise_for_status() # 检查请求是否成功
        soup_paper = BeautifulSoup(response_paper.text, 'html.parser')
        abstract = soup_paper.find('div', class_='abstract').text.strip()
        # 文本翻译
        article_chunk = split_text(abstract, max_length=1000)
        translated_text = str()
        for chunk in article_chunk:
            encoded_en = tokenizer(chunk, return_tensors="pt").to('cuda')
            generated_tokens = model.generate(
                **encoded_en,
                forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
            )
            ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            # print(ans[0])
            translated_text += ans[0]
        # print()
        # print(translated_text)
        
        # print(f'Title: {title}')
        # print(f'Title Translated: {title_chinese}')
        # print(f'Authors: {authors}')
        # print(f'Abstract: {abstract}')
        # print(f'Translated Abstract: {translated_text}')
        # # print(f'Authors: {authors}')
        paper_lists.append({
            "title": title,
            "title_chinese": title_chinese,
            "authors": authors,
            "abstract": abstract,
            "translated_abstract": translated_text,
            "paper_url": paper_url,
            "paper_pdf": paper_pdf
        })
        # print('-'*80)
        time.sleep(0.5)
    return paper_lists
    
if __name__ == "__main__":
    papers = fetch_icml_papers()
    num_papers = len(papers)
    print(f"Total papers: {num_papers}")
    # for paper in papers:
    #     print(f'Title: {paper["title"]}')
    #     print(f'Title Translated: {paper["title_chinese"]}')
    #     print(f'Authors: {paper["authors"]}')
    #     print(f'Abstract: {paper["abstract"]}')
    #     print(f'Translated Abstract: {paper["translated_abstract"]}')
    #     print(f'Paper URL: {paper["paper_url"]}')
    #     print(f'Paper PDF: {paper["paper_pdf"]}')
    #     print('-'*80)
    with open('CoRL_papers_2023.json', 'w') as f:
        json.dump(papers, f, indent=4, ensure_ascii=False)
    
    markdown_content = ""
    for paper in papers:
        markdown_content += f"# {paper['title']}\n"
        markdown_content += f"**题目:** {paper['title_chinese']}\n\n"
        markdown_content += f"**作者:** {paper['authors']}\n\n"
        markdown_content += f"**Abstract:** {paper['abstract']}\n\n"
        markdown_content += f"**摘要:** {paper['translated_abstract']}\n\n"

    with open('CoRL_2023_paper.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    