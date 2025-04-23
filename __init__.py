from .nodes import LoadDiaModel, InputDiaText, LoadDiaAudio, DiaTTS, SaveDiaAudio

NODE_CLASS_MAPPINGS = {
    "LoadDiaModel": LoadDiaModel,
    "InputDiaText": InputDiaText,
    "LoadDiaAudio": LoadDiaAudio,
    "DiaTTS": DiaTTS,
    "SaveDiaAudio": SaveDiaAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDiaModel": "Load Dia Model",
    "InputDiaText": "Input Dia Text",
    "LoadDiaAudio": "Load Dia Audio",
    "DiaTTS": "Dia TTS",
    "SaveDiaAudio": "Save Dia Audio",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
