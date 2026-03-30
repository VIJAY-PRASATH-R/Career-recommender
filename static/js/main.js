/**
 * CareerAI – main.js
 * Handles: drag-and-drop upload, form loading state, file name display
 */

document.addEventListener("DOMContentLoaded", () => {

  /* ── Drag-and-Drop Upload ──────────────────────────────────────── */
  const dropZone   = document.getElementById("dropZone");
  const fileInput  = document.getElementById("resume");
  const fileNameEl = document.getElementById("dropFileName");

  if (dropZone && fileInput) {
    // Show filename when a file is selected
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (file && fileNameEl) {
        fileNameEl.textContent = `✅ ${file.name}`;
        fileNameEl.style.display = "block";
      }
    });

    // Drag-over highlight
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("drag-over");
    });

    // Drop handler
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");

      const file = e.dataTransfer.files[0];
      if (file && file.type === "application/pdf") {
        // Push the dropped file into the hidden input via DataTransfer
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;

        if (fileNameEl) {
          fileNameEl.textContent = `✅ ${file.name}`;
          fileNameEl.style.display = "block";
        }
      } else {
        alert("Please drop a PDF file.");
      }
    });
  }

  /* ── Form Loading State ────────────────────────────────────────── */
  const form      = document.getElementById("analyzeForm");
  const submitBtn = document.getElementById("submitBtn");

  if (form && submitBtn) {
    form.addEventListener("submit", () => {
      const btnText    = submitBtn.querySelector(".btn-text");
      const btnLoading = submitBtn.querySelector(".btn-loading");

      if (btnText)    btnText.style.display    = "none";
      if (btnLoading) btnLoading.style.display = "flex";
      submitBtn.disabled = true;
    });
  }

  /* ── result page: animate progress bars on scroll ─────────────── */
  const bars = document.querySelectorAll(".progress-bar");
  if (bars.length) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.animationPlayState = "running";
        }
      });
    }, { threshold: 0.2 });

    bars.forEach((bar) => {
      bar.style.animationPlayState = "paused";
      observer.observe(bar);
    });
  }

});
