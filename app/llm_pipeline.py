from rag_retriever import retrieve
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_optimized_content(user_query):
    # 1. Retrieve context
    results = retrieve(user_query, top_k=3)

    context = "\n\n".join([r["text"] for r in results])

    # 2. Build prompt
prompt = f"""
You are an expert supplement content writer.

Write an SEO-optimised product description using ONLY the retrieved context below.

Strict rules:
- Do NOT add any fact that is not explicitly written in the context
- Do NOT invent ingredients, benefits, certifications, dosage details, or safety claims
- If some information is missing, leave it out
- Keep the wording clear, natural, and customer-friendly
- Keep claims cautious and non-medical
- Output only the product description

Retrieved context:
{context}

User request:
{user_query}
"""

    # 3. Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate safe, SEO-optimised supplement content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    output = generate_optimized_content("melatonin supplement for sleep")
    print(output)