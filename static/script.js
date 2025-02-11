document.addEventListener("DOMContentLoaded", function () {
    let fileInput = document.querySelector("input[type='file']");
    let button = document.querySelector("button");

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            button.style.backgroundColor = "#008CBA";
        } else {
            button.style.backgroundColor = "#ff9800";
        }
    });

    button.addEventListener("mouseenter", function () {
        button.style.opacity = 0.8;
    });

    button.addEventListener("mouseleave", function () {
        button.style.opacity = 1;
    });
});
