from config.settings import Settings

def test_default_settings():
    settings = Settings(openai_api_key="test")
    assert settings.openai_model == "gpt-4.1"
    assert settings.app_name == "SemanticScout"
