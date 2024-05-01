document.addEventListener("mousemove", function (e) {
    const x = e.clientX / window.innerWidth - 0.5;
    const y = e.clientY / window.innerHeight - 0.5;
  
    const background = document.querySelector(".background");
    const scale = 1.5;
  
    background.style.background = `radial-gradient(circle at ${x * 100}% ${y * 100}%, #ff6b6b, #3a1c71)`;
    background.style.transform = `scale(${scale})`;
  });
  