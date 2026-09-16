from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("detection", "0002_evaluationmetrics_roc_auc_frameheatmap_heatmap_png_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysissession",
            name="media_kind",
            field=models.CharField(
                choices=[
                    ("video", "Video"),
                    ("image", "Image"),
                    ("audio", "Audio"),
                ],
                default="video",
                max_length=8,
            ),
        ),
    ]
