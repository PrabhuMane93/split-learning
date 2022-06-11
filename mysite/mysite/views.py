from django.shortcuts import render, HttpResponse
import sys
from subprocess import run,PIPE
import os

def index(request):
    return render(request, template_name='index.html')

def ready(request):
    return render(request, template_name='ready.html')

def startml(request):
    inp= request.POST.get('path')
    out= run([sys.executable,'D://Projects//SplitDjango//mysite//mysite//ml//h.py',inp],stdout=PIPE)
    print(out)
    os.system('pyinstaller --onefile -w temp.py')
    return render(request,'index.html',{'data1':out})

def images(request):
    return render(request, template_name='images.html')