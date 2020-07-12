import requests

def lineNotify(message):
    payload = {'message':message}
    return _lineNotify(payload)

def notifyFile(filename):
    file = {'imageFile':open(filename,'rb')}
    payload = {'message': 'test'}
    return _lineNotify(payload,file)

def notifyPicture(url):
    payload = {'message':" ",'imageThumbnail':url,'imageFullsize':url}
    return _lineNotify(payload)

def notifySticker(stickerID,stickerPackageID):
    payload = {'message':" ",'stickerPackageId':stickerPackageID,'stickerId':stickerID}
    return _lineNotify(payload)

def _lineNotify(payload,file=None):
    import requests
    url = 'https://notify-api.line.me/api/notify'
    #token = 'iOlY60l0cojfvypypS6xC0ln3X0b5TtFKz1jUTryDl0'	#EDIT
    token = '8Mk2ufXnQDV97D3wFSLGHTGVGvtSz3Pu7MiqG8jOw9V'
    headers = {'Authorization':'Bearer '+token}
    return requests.post(url, headers=headers , data = payload, files=file)

# lineNotify('ทดสอบภาษาไทย hello')
notifySticker(40,2)
#notifyPicture("https://static.naewna.com/uploads/news/source/456101.jpg")
