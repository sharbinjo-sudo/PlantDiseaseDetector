from django.db import models
from django.conf import settings


class PredictionRecord(models.Model):
    """
    Stores each image uploaded from Flutter and the model prediction result.
    Works with Cloudinary (Render/Neon) or local storage (dev).
    """

    image = models.ImageField(
        upload_to="predictions/",
        help_text="Uploaded image (stored locally in dev, Cloudinary in prod)",
    )

    predicted_label = models.CharField(max_length=255)
    confidence = models.FloatField(help_text="Confidence score (0.0 - 1.0)")

    model_type = models.CharField(max_length=50, default="CNN")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Prediction Record"
        verbose_name_plural = "Prediction Records"

    def __str__(self):
        return f"{self.predicted_label} ({self.confidence * 100:.2f}%)"

    def image_url(self):
        """
        Return full image URL (Cloudinary or local /media/ path).
        """
        if self.image and hasattr(self.image, "url"):
            return self.image.url
        return None

    def image_filename(self):
        """
        Return only the filename from the path (useful in admin UI).
        """
        return self.image.name.split("/")[-1] if self.image else None
