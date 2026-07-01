// Click-to-expand lightbox shared by all lab-notes pages. Clicking a plot
// opens it full-size over a dark overlay; click anywhere or press Escape to
// close. Kept deliberately plain so non-JS-specialists can follow it.

const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxCaption = document.getElementById("lightbox-caption");

function openLightbox(imageSrc, captionText) {
  lightboxImg.src = imageSrc;
  lightboxImg.alt = captionText;
  lightboxCaption.textContent = captionText;
  lightbox.classList.add("open");
}

function closeLightbox() {
  lightbox.classList.remove("open");
  lightboxImg.src = "";
}

// Wire up every plot image on the page.
const plotImages = document.querySelectorAll(".plot img");
plotImages.forEach(function (image) {
  image.addEventListener("click", function () {
    const figure = image.closest("figure");
    const caption = figure.querySelector("figcaption");
    const captionText = caption ? caption.textContent.trim() : "";
    openLightbox(image.src, captionText);
  });
});

// Close on background click or Escape.
lightbox.addEventListener("click", closeLightbox);
document.addEventListener("keydown", function (event) {
  if (event.key === "Escape") {
    closeLightbox();
  }
});
