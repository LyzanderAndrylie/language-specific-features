import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4.1"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = """
You are an expert linguistic evaluator specializing in analyzing "steered generated text." Your task is to assess a given text input based on a strict two-part rubric: Language Occurrence and Language Coherence. Please analyze the input text based on the following rubric and provide the output strictly adhering to the schema defined below.

## Text Generation Assessment Rubric

This rubric is designed to be applied manually by humans or automatically with an LLM, and aims to ensure consistency and transparency in evaluating whether and how language-specific features affect generated outputs.

### Language Occurrence

This criterion captures whether a meaningful language change has occurred in the generated text. A change is considered to have occurred if the output contains at least one morpheme (the smallest meaningful unit of language) from a different language than English.

- Score 0 (Unchange): The text remains fully in English and contains no morphemes from another language. Minor orthographic changes that do not introduce new meaning or reflect actual cross-linguistic variation are not counted as a language change. For example:
    - apple -> åpple (still English, only orthographic variation)
    - naive -> naïve (still English usage)

- Score 1 (Change): At least one morpheme from a different language is introduced in the output, even if the change occurs only in part of the sentence. For example:
    - apple -> manzana (Spanish)
    - He is a teacher -> Er ist ein Lehrer (German)

### Language Coherence

This criterion evaluates the overall intelligibility and fluency of the generated text when steered toward multilingual expression. It is assessed along a three-point ordinal scale:

- Incoherent: The text is unintelligible and does not form meaningful sentences. It may consist of repeated characters, morphemes, or nonsensical word sequences. For example:
    - blblblblbl èèèè $$$$$
    - rerere banana moon qwerty

- Partially Coherent: The general meaning of the text can be inferred, but parts may require guessing or interpretation. This includes major grammatical or syntactic errors, repetitive yet intelligible structures, or repetitions at the end of the text. For examples:
    - He go to escuela because want aprender
    - The cat is sleep sleep the bed the bed

- Coherent: The text is readable and understandable, with only minor issues (e.g., minor grammatical or syntactic errors and mid-sentence code-switching that does not obscure meaning). Code-switching is acceptable as long as overall sentence-level coherence is maintained. For examples:
    - She goes to the bibliothèque every evening to study
    - I like to eat sushi y también tacos
    - Er ist ein Lehrer und he teaches German very well
"""

USER_PROMPT = """
## Input

{text}
"""


class LLMCheckResult(BaseModel):
    language_occurrence_score: int = Field(
        description="0 for no language change, 1 for language change"
    )
    language_coherence_score: int = Field(
        description="0 for incoherent, 1 for partially coherent, 2 for coherent"
    )


current_dir = Path(__file__).parent
data_dir = current_dir.parent / "all"
langs = [
    "eng_Latn",
    # "deu_Latn",
    "fra_Latn",
    "ita_Latn",
    "por_Latn",
    "hin_Deva",
    "spa_Latn",
    "tha_Thai",
    "bul_Cyrl",
    "rus_Cyrl",
    "tur_Latn",
    "vie_Latn",
    "jpn_Jpan",
    "kor_Hang",
    "cmn_Hans",
]


def call_openrouter(input_text: str, max_retries: int = 3) -> LLMCheckResult:
    for attempt in range(1, max_retries + 1):
        try:
            prompt = USER_PROMPT.format(text=input_text)
            completion = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=LLMCheckResult,
                temperature=0,
                max_tokens=100,
            )
            result = completion.choices[0].message.parsed
            return result

        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                backoff = 2 - attempt
                print(f"Retrying in {backoff}s...")
                time.sleep(backoff)

    raise RuntimeError(f"OpenRouter API call failed after {max_retries} attempts")


def llm_check(row: pd.Series) -> pd.Series:
    input_text = row.get("generated_text")

    try:
        result = call_openrouter(input_text)

        return pd.Series(
            {
                "occurrence": result.language_occurrence_score,
                "coherence": result.language_coherence_score,
            }
        )

    except RuntimeError as e:
        print(f"Error processing row: {e}")

        return pd.Series({"occurrence": -1, "coherence": -1})


def main():
    for lang in langs:
        subdir = data_dir / lang

        for file_path in subdir.glob("*.csv"):
            print(f"Processing {file_path.name}...")

            output_path = current_dir / lang / file_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df = pd.read_csv(file_path)
            eval_results = df.apply(llm_check, axis=1)
            df = pd.concat([df, eval_results], axis=1)

            df.to_csv(output_path, index=False)
            print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
