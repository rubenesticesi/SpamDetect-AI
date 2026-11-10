from app.app import app

def test_home():
    client=app.test_client()
    response=client.get('/')
    assert response.status_code==200
    assert b'SpamDetect AI' in response.data

def test_blank():
    client=app.test_client()
    response=client.post('/',data={'message':'  '})
    assert b'Introduce un mensaje' in response.data
