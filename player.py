from pathlib import Path
import streamlit.components.v1 as components
from local_media import media_url

component=components.declare_component('video_boundary_editor_v8',path=str(Path(__file__).parent/'player'))
montage_component=components.declare_component('montage_timeline_v7',path=str(Path(__file__).parent/'timeline'))


def editor(source,start,end,duration,key,theme='light',manual=False,ack=None):
    return component(url=media_url(source) if source else '',start=start,end=end,duration=duration,
                     app_theme=theme,manual=manual,ack=ack,key=key,default=None)

def timeline(clips,theme='light',key='montage-timeline'):
    data=[]
    for item in clips:
        data.append({k:v for k,v in item.items() if k != 'source'})
        data[-1]['url']=media_url(item['source']) if item.get('source') else ''
    return montage_component(clips=data,app_theme=theme,key=key,default=None)
