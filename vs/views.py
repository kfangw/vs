from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage

from vs.main import vs


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        # fs.save(myfile.name, myfile)
        filename = fs.save(myfile.name, myfile)
        print "{}/{}".format(settings.MEDIA_ROOT, filename)
        result = vs(["--anchor", "{}/{}".format(settings.MEDIA_ROOT, filename), settings.MEDIA_ROOT, ])
        result_json = []
        for r in result:
            if r[0] and isinstance(r[0], basestring):
                result_json.append(
                    {
                        "target": r[0].replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
                        "score": r[1]
                    }
                )

        return JsonResponse({'success': 'ok', "data": result_json})
    return JsonResponse({'success': 'ok'})
