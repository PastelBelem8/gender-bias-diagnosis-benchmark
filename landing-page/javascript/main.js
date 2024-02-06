let lastScrollTop = 0;

window.addEventListener("scroll", function() {
  let currentScroll = window.pageYOffset || document.documentElement.scrollTop;

  if (currentScroll > lastScrollTop) {
    // Scroll down
    document.getElementById("header").style.top = "-60px"; // Hides the header
  } else {
    // Scroll up
    document.getElementById("header").style.top = "0"; // Shows the header
  }
  lastScrollTop = currentScroll;
});
