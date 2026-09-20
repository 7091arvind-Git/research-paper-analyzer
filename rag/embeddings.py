import time
from google import genai
import config

def get_client(api_key=None):
    key = api_key or config.GOOGLE_API_KEY
    return genai.Client(api_key=key)

def get_embedding(text, client=None, api_key=None):
    if client is None:
        client = get_client(api_key)
    models = getattr(config, "FALLBACK_EMBEDDING_MODELS", [config.EMBEDDING_MODEL])
    last_error = None
    for model_name in models:
        for attempt in range(2):
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=text
                )
                return response.embeddings[0].values
            except Exception as e:
                last_error = e
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 1:
                    time.sleep(2)
                else:
                    break
    raise last_error

def get_embeddings_batch(texts, client=None, api_key=None, batch_size=25):
    if client is None:
        client = get_client(api_key)
    models = getattr(config, "FALLBACK_EMBEDDING_MODELS", [config.EMBEDDING_MODEL])
    last_error = None
    for model_name in models:
        try:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                contents = [{"parts": [{"text": t}]} for t in batch]
                for attempt in range(2):
                    try:
                        response = client.models.embed_content(
                            model=model_name,
                            contents=contents
                        )
                        for emb in response.embeddings:
                            embeddings.append(emb.values)
                        break
                    except Exception as e:
                        if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 1:
                            time.sleep(3)
                        else:
                            raise e
            return embeddings
        except Exception as e:
            last_error = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            else:
                raise e
    raise last_error
