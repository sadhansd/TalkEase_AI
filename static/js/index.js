function changeSlide1(direction) {
        const selectElement = document.getElementById("selectCarousel1");
        const selectedIndex = selectElement.selectedIndex;

        const carousel = document.getElementById("carousel1");
        const itemWidth = document.querySelector(".carousel-item").offsetWidth;

        if (direction === "prev" && selectedIndex > 0) {
          selectElement.selectedIndex = selectedIndex - 1;
        } else if (
          direction === "next" &&
          selectedIndex < selectElement.options.length - 1
        ) {
          selectElement.selectedIndex = selectedIndex + 1;
        }

        carousel.style.transform = `translateX(${
          -selectElement.selectedIndex * itemWidth
        }px)`;
      }

      function changeSlide2(direction) {
        const selectElement = document.getElementById("selectCarousel2");
        const selectedIndex = selectElement.selectedIndex;

        const carousel = document.getElementById("carousel2");
        const itemWidth = document.querySelector(".carousel-item").offsetWidth;

        if (direction === "prev" && selectedIndex > 0) {
          selectElement.selectedIndex = selectedIndex - 1;
        } else if (
          direction === "next" &&
          selectedIndex < selectElement.options.length - 1
        ) {
          selectElement.selectedIndex = selectedIndex + 1;
        }

        carousel.style.transform = `translateX(${
          -selectElement.selectedIndex * itemWidth
        }px)`;
      }


      function checkAndShowAlert() {
            var selectedOption1 = document.getElementById('selectCarousel1').value;
            var selectedOption2 = document.getElementById('selectCarousel2').value;

            // Check the selected option and show the alert conditionally
            if (selectedOption1 === selectedOption2) {
                var alertMessage = "Please select different options";
                alert(alertMessage);
            }
        }

