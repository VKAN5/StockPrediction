$(document).ready(function(){

  //Initialize with the list of symbols
  let names = ["AAPL","ADBE","ATVI","GOOG","MSFT","AMZN","AMD","NVDA","COST",
               "TSLA","GOOGL","NFLX","SBUX","CSCO", "INTC"]

  //Find the input search box
  let search = document.getElementById("searchCoin")

  //Find every item inside the dropdown
  let items = document.getElementsByClassName("dropdown-item")
  function buildDropDown(values) {
      let contents = []
      for (let name of values) {
      contents.push('<input type="button" class="dropdown-item" type="button" value="' + name + '"/>')
      }
      $('#menuItems').append(contents.join(""))

      //Hide the row that shows no items were found
      $('#empty').hide()
  }

  //Capture the event when user types into the search box
window.addEventListener('input', function () {
    filter(search.value.trim().toLowerCase())
})

//For every word entered by the user, check if the symbol starts with that word
//If it does show the symbol, else hide it
function filter(word) {
    let length = items.length
    let collection = []
    let hidden = 0
    for (let i = 0; i < length; i++) {
    if (items[i].value.toLowerCase().startsWith(word)) {
        $(items[i]).show()
    }
    else {
        $(items[i]).hide()
        hidden++
   }
   }

   //If all items are hidden, show the empty view
   if (hidden === length) {
   $('#empty').show()
   }
   else {
   $('#empty').hide()
   }
}


//If the user clicks on any item, set the title of the button as the text of the item
$('#menuItems').on('click', '.dropdown-item', function(){
   $('#dropdown_coins1').text($(this)[0].value)
   $("#dropdown_coins1").dropdown('toggle');
})
buildDropDown(names)

// //Find the input search box
// let search2 = document.getElementById("searchCoin2")
//
// //Find every item inside the dropdown
// items2 = document.getElementsByClassName("dropdown-item2")
// function buildDropDown2(values) {
//     let contents = []
//     for (let name of values) {
//     contents.push('<input type="button" class="dropdown-item2" type="button2" value="' + name + '"/>')
//     }
//     $('#menuItems2').append(contents.join(""))
//
//     //Hide the row that shows no items were found
//     $('#empty2').hide()
// }
//
// //Capture the event when user types into the search box
// window.addEventListener('input', function () {
//   filter(search.value.trim().toLowerCase())
// })
//
// //For every word entered by the user, check if the symbol starts with that word
// //If it does show the symbol, else hide it
// function filter(word) {
//   let length = items2.length
//   let collection = []
//   let hidden = 0
//   for (let i = 0; i < length; i++) {
//   if (items2[i].value.toLowerCase().startsWith(word)) {
//       $(items2[i]).show()
//   }
//   else {
//       $(items2[i]).hide()
//       hidden++
//  }
//  }
//
//  //If all items are hidden, show the empty view
//  if (hidden === length) {
//  $('#empty2').show()
//  }
//  else {
//  $('#empty2').hide()
//  }
// }
//
//
// $('#menuItems2').on('click', '.dropdown-item2', function(){
//    $('#dropdown_coins2').text($(this)[0].value)
//    $("#dropdown_coins2").dropdown('toggle');
// })
//
// buildDropDown2(names)



$('#btn-predict-linear').click(function(){
  console.log('Vijay');
  console.log($('#dropdown_coins1').text());
  var data = $('#dropdown_coins1').text();
  console.log(data);

$('.loader1').show();

$.ajax({
  type: 'POST',
  url: '/predict',
  data: data,
  contentType: false,
  cache:false,
  processData:false,
  async:true,
  success:function (op_data) {
    $('.loader1').hide();
    document.getElementById("loadresult").innerHTML = op_data;
    console.log(op_data);
  }
});

});

$('#btn-predict-quad').click(function(){
  console.log('Vijay');
  var data = $('#dropdown_coins1').text();
  console.log(data);

$('.loader2').show();
$.ajax({
  type: 'POST',
  url: '/predictquad',
  data: data,
  contentType: false,
  cache:false,
  processData:false,
  async:true,
  success:function (op2_data) {
    $('.loader2').hide();
    document.getElementById("loadresultquad").innerHTML = op2_data;
    console.log(op2_data);
  }
});


});

$('#btn-predict-knn').click(function(){

  var data = $('#dropdown_coins1').text();
  console.log(data);
  $('.loader3').show();


$.ajax({
  type: 'POST',
  url: '/predictknn',
  data: data,
  contentType: false,
  cache:false,
  processData:false,
  async:true,
  success:function (op3_data) {
    $('.loader3').hide();
    document.getElementById("loadresultknn").innerHTML = op3_data;
    console.log(op3_data);
  }
});


});


});
