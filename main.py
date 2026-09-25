from fastapi import FastAPI, Response

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/voice")
async def voice_endpoint():
    # Build your raw TwiML string
    twiml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say>Hello! Thank you for calling.</Say>
        </Response>
        """
    
    # Return a custom Response with the application/xml media type
    return Response(content=twiml_content, media_type="application/xml")