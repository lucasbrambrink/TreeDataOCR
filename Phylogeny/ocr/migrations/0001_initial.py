# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ImageAnalysis',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('data_source', models.CharField(max_length=100)),
                ('image', models.ImageField(upload_to=b'')),
                ('processed_polygon', models.TextField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
