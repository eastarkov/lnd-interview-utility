"""Validated, durable selections and order; independent of the active search."""
import hashlib


def clip_id(record_key,start,end):
    return hashlib.sha256(f'{record_key}:{start:.4f}:{end:.4f}'.encode()).hexdigest()[:24]


def source_duration(record):
    return record.get('media_duration') or record.get('duration') or 0


def clip_bounds(start, end, duration):
    start, end = float(start), float(end)
    if not (0 <= start < min(end, duration) and end <= duration + .001):
        raise ValueError('Границы фрагмента должны быть внутри исходного видео.')
    return start, min(end, duration)


def validate(items,records):
    result=[];seen=set()
    for item in items:
        try:
            key=item['record_key'];record=records[key]
            start, end = clip_bounds(item['start'], item['end'], source_duration(record))
            if item['id'] in seen:
                continue
            seen.add(item['id'])
            result.append(dict(item,start=start,end=end))
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
        duration = source_duration(records[existing['record_key']])
        start, end = clip_bounds(action['start'], action['end'], duration)
        return [dict(item,start=start,end=end) if item['id']==target else item for item in items]
    raise ValueError('Неизвестное действие монтажа.')
