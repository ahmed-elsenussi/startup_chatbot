import json
import textwrap
from django.core.management.base import BaseCommand
from chatbot.models import Company

class Command(BaseCommand):
    help = "Export company data into JSON with RAG-ready text chunks"

    def handle(self, *args, **kwargs):
        data = []

        for company in Company.objects.all():
            # Base text
            description = company.description or ""
            
            # Remove repeated company name at start
            if description.startswith(company.name or ""):
                description = description[len(company.name):].strip()

            text = f"{company.name or 'This company'} operates in Culture. {description}"

            # Split into chunks (~500 chars each)
            chunks = textwrap.wrap(text, width=500)

            for i, chunk in enumerate(chunks):
                data.append({
                    "id": company.id,
                    "text": chunk,
                    "logo_image": company.logo_image or ""
                })

        # Save to JSON
        with open("company_chunks.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.stdout.write(self.style.SUCCESS(
            f"Exported {len(data)} chunks to company_chunks.json"
        ))
