// Open the book's figures at full size when they are clicked.
// Jupyter Book adds every .js file in _static to each page.
document.addEventListener("DOMContentLoaded", function () {
  var selector = "main figure img, main .cell_output img, main img.align-center";
  var images = document.querySelectorAll(selector);

  function close(overlay) {
    overlay.remove();
    document.removeEventListener("keydown", overlay._onKey);
  }

  images.forEach(function (img) {
    // Leave images that are already links alone
    if (img.closest("a")) {
      return;
    }

    img.classList.add("zoomable");
    img.addEventListener("click", function () {
      var overlay = document.createElement("div");
      overlay.className = "figure-zoom-overlay";
      overlay.setAttribute("role", "dialog");
      overlay.setAttribute("aria-label", img.alt || "Figure");

      var large = document.createElement("img");
      large.src = img.currentSrc || img.src;
      large.alt = img.alt;
      overlay.appendChild(large);

      overlay._onKey = function (event) {
        if (event.key === "Escape") {
          close(overlay);
        }
      };
      overlay.addEventListener("click", function () {
        close(overlay);
      });
      document.addEventListener("keydown", overlay._onKey);
      document.body.appendChild(overlay);
    });
  });
});
