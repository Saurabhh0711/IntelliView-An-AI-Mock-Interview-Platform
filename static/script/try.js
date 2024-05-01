document.addEventListener("DOMContentLoaded", function() {
    // Remove the loading class from body after 3 seconds
    setTimeout(function() {
      document.body.classList.remove('loading');
    }, 3000);
  });
  