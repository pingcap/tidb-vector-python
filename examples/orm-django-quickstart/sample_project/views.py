from django.http import HttpResponse, JsonResponse
from django_tidb.fields.vector import CosineDistance

from .models import Document


# Insert 3 documents.
def insert_documents(request):
    Document.objects.create(content="dog", embedding=[1, 2, 1])
    Document.objects.create(content="fish", embedding=[1, 2, 4])
    Document.objects.create(content="tree", embedding=[1, 0, 0])

    return HttpResponse("Insert documents successfully.")


# Get 3-nearest neighbor documents.
def get_nearest_neighbors_documents(request):
    results = Document.objects.annotate(
        distance=CosineDistance('embedding', [1, 2, 3])
    ).order_by('distance')[:3]
    response = []
    for doc in results:
        response.append({
            'distance': doc.distance,
            'document': doc.content
        })

    return JsonResponse(response, safe=False)


# Get documents within a certain distance.
def get_documents_within_distance(request):
    results = Document.objects.annotate(
        distance=CosineDistance('embedding', [1, 2, 3])
    ).filter(distance__lt=0.2).order_by('distance')[:3]
    response = []
    for doc in results:
        response.append({
            'distance': doc.distance,
            'document': doc.content
        })

    return JsonResponse(response, safe=False)


def list_routes(request):
    return JsonResponse({
        'routes': [
            '/insert_documents',
            '/get_nearest_neighbors_documents',
            '/get_documents_within_distance'
        ]
    })
