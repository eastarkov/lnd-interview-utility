"""Validated, durable selections and order; independent of the active search."""
import hashlib


def clip_id(record_key,start,end):
    return hashlib.sha256(f'{record_key}:{start:.4f}:{end:.4f}'.encode()).hexdigest()[:24]


def validate(items,records):
    result=[];seen=set()
    for item in items:
        try:
            key=item['record_key'];record=records[key]
            start,end=float(item['start']),float(item['end'])
            if record.get('pending') or not 0<=start<end<=record['duration']+.001 or item['id'] in seen:
                continue
            seen.add(item['id'])
            result.append(dict(item,start=start,end=min(end,record['duration'])))
        except (KeyError,ValueError,TypeError):
            continue
    return result


def reordered(items,order):
    by_id={item['id']:item for item in items}
    if len(order)!=len(by_id) or len(set(order))!=len(order) or set(order)!=set(by_id):
        raise ValueError('Некорректный порядок фрагментов.')
    return [by_id[key] for key in order]


def edited(items, action, records):
    """Apply one validated trim/remove event without recreating deselected clips."""
    target=action.get('id')
    existing=next((item for item in items if item['id']==target),None)
    if existing is None:
        return list(items)
    if action.get('type')=='remove':
        return [item for item in items if item['id']!=target]
    if action.get('type')=='trim':
        start,end=float(action['start']),float(action['end'])
        duration=records[existing['record_key']]['duration']
        if not 0<=start<end<=duration+.001:
            raise ValueError('Границы фрагмента должны быть внутри исходного видео.')
        return [dict(item,start=start,end=min(end,duration)) if item['id']==target else item for item in items]
    raise ValueError('Неизвестное действие монтажа.')
