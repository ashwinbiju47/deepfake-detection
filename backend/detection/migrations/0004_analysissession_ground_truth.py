from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("detection", "0003_analysissession_media_kind"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysissession",
            name="ground_truth",
            field=models.CharField(
                blank=True,
                choices=[("real", "Real"), ("fake", "Fake")],
                db_index=True,
                max_length=8,
                null=True,
            ),
        ),
    ]
