
import hashlib
from io import BytesIO

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI


class RAGEngine:

    def __init__(self, api_key):

        self.ai = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        self.model_name = "openai/gpt-oss-120b"

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.db = chromadb.PersistentClient(
            path="./chromadb.db"
        )

        self.collection = self.db.get_or_create_collection(
            name="my_pdf"
        )

    # PDF faylını oxuyur və hissələrə ayırır
    def process_pdf(self, pdf_bytes):

        pdf = PdfReader(BytesIO(pdf_bytes))

        chunks = []
        metadatas = []
        ids = []

        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text()

            if not text:
                continue

            chunk_size = 1000

            for i in range(0, len(text), chunk_size):

                chunk = text[i:i + chunk_size]

                chunks.append(chunk)

                metadatas.append({
                    "page": page_number
                })

                chunk_id = hashlib.sha256(
                    f"{page_number}-{i}-{chunk}".encode()
                ).hexdigest()

                ids.append(chunk_id)

        if not chunks:
            raise ValueError("PDF-dən mətn oxunmadı.")

        embeddings = self.embedding_model.encode(
            chunks
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(chunks)

    # Suala uyğun PDF hissələrini tapır
    def search(self, question):

        question_embedding = self.embedding_model.encode(
            [question]
        ).tolist()

        results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=3
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        pages = list(set(
            metadata["page"] for metadata in metadatas
        ))

        return documents, pages

    # AI vasitəsilə cavab hazırlayır
    def ask(self, question):

        documents, pages = self.search(question)

        context = "\n".join(documents)

        response = self.ai.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Yalnız verilən PDF məlumatlarına əsasən cavab ver. Azərbaycan dilində yaz."
                },
                {
                    "role": "user",
                    "content": f"""
PDF məlumatları:
{context}

Sual:
{question}
"""
                }
            ]
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": pages
        }
