from pydantic_settings import BaseSettings

class VapiSettings(BaseSettings):
    vapi_api_key: str | None = None
    vapi_api_url: str = "https://api.vapi.ai"
    vapi_public_key: str | None = None
    vapi_assistant_id: str | None = None
    cohere_api_key: str | None = None
    pinecone_api_key: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"