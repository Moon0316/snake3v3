(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["example-create~example-edit"],{1172:function(t,e,a){"use strict";var r=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"createPost-container"},[a("el-form",{ref:"postForm",staticClass:"form-container",attrs:{model:t.postForm,rules:t.rules}},[a("sticky",{attrs:{"z-index":10,"class-name":"sub-navbar "+t.postForm.status}},[a("comment-dropdown",{model:{value:t.postForm.disableComment,callback:function(e){t.$set(t.postForm,"disableComment",e)},expression:"postForm.disableComment"}}),a("platform-dropdown",{model:{value:t.postForm.platforms,callback:function(e){t.$set(t.postForm,"platforms",e)},expression:"postForm.platforms"}}),a("source-url-dropdown",{model:{value:t.postForm.sourceURL,callback:function(e){t.$set(t.postForm,"sourceURL",e)},expression:"postForm.sourceURL"}}),a("el-button",{directives:[{name:"loading",rawName:"v-loading",value:t.loading,expression:"loading"}],staticStyle:{"margin-left":"10px"},attrs:{type:"success"},on:{click:t.submitForm}},[t._v(" Publish ")]),a("el-button",{directives:[{name:"loading",rawName:"v-loading",value:t.loading,expression:"loading"}],attrs:{type:"warning"},on:{click:t.draftForm}},[t._v(" Draft ")])],1),a("div",{staticClass:"createPost-main-container"},[a("el-row",[a("warning"),a("el-col",{attrs:{span:24}},[a("el-form-item",{staticStyle:{"margin-bottom":"40px"},attrs:{prop:"title"}},[a("material-input",{attrs:{maxlength:100,name:"name",required:""},model:{value:t.postForm.title,callback:function(e){t.$set(t.postForm,"title",e)},expression:"postForm.title"}},[t._v(" Title ")])],1),a("div",{staticClass:"postInfo-container"},[a("el-row",[a("el-col",{attrs:{span:8}},[a("el-form-item",{staticClass:"postInfo-container-item",attrs:{"label-width":"60px",label:"Author:"}},[a("el-select",{attrs:{"remote-method":t.getRemoteUserList,filterable:"","default-first-option":"",remote:"",placeholder:"Search user"},model:{value:t.postForm.author,callback:function(e){t.$set(t.postForm,"author",e)},expression:"postForm.author"}},t._l(t.userListOptions,(function(t,e){return a("el-option",{key:t+e,attrs:{label:t,value:t}})})),1)],1)],1),a("el-col",{attrs:{span:10}},[a("el-form-item",{staticClass:"postInfo-container-item",attrs:{"label-width":"120px",label:"Publish Time:"}},[a("el-date-picker",{attrs:{type:"datetime",format:"yyyy-MM-dd HH:mm:ss",placeholder:"Select date and time"},model:{value:t.timestamp,callback:function(e){t.timestamp=e},expression:"timestamp"}})],1)],1),a("el-col",{attrs:{span:6}},[a("el-form-item",{staticClass:"postInfo-container-item",attrs:{"label-width":"90px",label:"Importance:"}},[a("el-rate",{staticStyle:{display:"inline-block"},attrs:{max:3,colors:["#99A9BF","#F7BA2A","#FF9900"],"low-threshold":1,"high-threshold":3},model:{value:t.postForm.importance,callback:function(e){t.$set(t.postForm,"importance",e)},expression:"postForm.importance"}})],1)],1)],1)],1)],1)],1),a("el-form-item",{staticStyle:{"margin-bottom":"40px"},attrs:{"label-width":"70px",label:"Summary:"}},[a("el-input",{staticClass:"article-textarea",attrs:{rows:1,type:"textarea",autosize:"",placeholder:"Please enter the content"},model:{value:t.postForm.abstractContent,callback:function(e){t.$set(t.postForm,"abstractContent",e)},expression:"postForm.abstractContent"}}),a("span",{directives:[{name:"show",rawName:"v-show",value:t.abstractContentLength,expression:"abstractContentLength"}],staticClass:"word-counter"},[t._v(t._s(t.abstractContentLength)+"words")])],1),a("el-form-item",{staticStyle:{"margin-bottom":"30px"},attrs:{prop:"content"}},[t.tinymceActive?a("tinymce",{ref:"editor",attrs:{height:400},model:{value:t.postForm.fullContent,callback:function(e){t.$set(t.postForm,"fullContent",e)},expression:"postForm.fullContent"}}):t._e()],1),a("el-form-item",{staticStyle:{"margin-bottom":"30px"},attrs:{prop:"imageURL"}},[a("upload-image",{model:{value:t.postForm.imageURL,callback:function(e){t.$set(t.postForm,"imageURL",e)},expression:"postForm.imageURL"}})],1)],1)],1)],1)},o=[],n=(a("99af"),a("d81d"),a("b0c0"),a("96cf"),a("1da1")),s=a("d4ec"),i=a("bee2"),l=a("262e"),c=a("2caf"),u=a("9ab4"),m=a("1b40"),p=a("75fb"),d=a("9d25"),f=a("b8f0"),b=a("ac1a"),v=a("e741"),h=a("da80"),g=a("b804"),w=a("8256"),y=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"upload-container"},[a("el-upload",{staticClass:"image-uploader",attrs:{data:t.dataObj,multiple:!1,"show-file-list":!1,"on-success":t.handleImageSuccess,drag:"",action:"https://httpbin.org/post"}},[a("i",{staticClass:"el-icon-upload"}),a("div",{staticClass:"el-upload__text"},[t._v(" 将文件拖到此处，或"),a("em",[t._v("点击上传")])])]),a("div",{staticClass:"image-preview image-app-preview"},[a("div",{directives:[{name:"show",rawName:"v-show",value:t.imageUrl.length>1,expression:"imageUrl.length>1"}],staticClass:"image-preview-wrapper"},[a("img",{attrs:{src:t.imageUrl}}),a("div",{staticClass:"image-preview-action"},[a("i",{staticClass:"el-icon-delete",on:{click:t.rmImage}})])])]),a("div",{staticClass:"image-preview"},[a("div",{directives:[{name:"show",rawName:"v-show",value:t.imageUrl.length>1,expression:"imageUrl.length>1"}],staticClass:"image-preview-wrapper"},[a("img",{attrs:{src:t.imageUrl}}),a("div",{staticClass:"image-preview-action"},[a("i",{staticClass:"el-icon-delete",on:{click:t.rmImage}})])])])],1)},O=[],j=function(t){Object(l["a"])(a,t);var e=Object(c["a"])(a);function a(){var t;return Object(s["a"])(this,a),t=e.apply(this,arguments),t.tempUrl="",t.dataObj={token:"",key:""},t}return Object(i["a"])(a,[{key:"emitInput",value:function(t){this.$emit("input",t)}},{key:"rmImage",value:function(){this.emitInput("")}},{key:"handleImageSuccess",value:function(t){this.emitInput(t.files.file)}},{key:"imageUrl",get:function(){return this.value}}]),a}(m["c"]);Object(u["a"])([Object(m["b"])({default:""})],j.prototype,"value",void 0),j=Object(u["a"])([Object(m["a"])({name:"UploadImage"})],j);var k=j,x=k,C=(a("c4fc"),a("2877")),F=Object(C["a"])(x,y,O,!1,null,"5b67becd",null),_=F.exports,U=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("aside",[t._v(" "+t._s(t.$t("example.warning"))+" "),a("a",{attrs:{href:"https://armour.github.io/vue-typescript-admin-docs/guide/essentials/tags-view.html",target:"_blank"}},[t._v("Document")])])},L=[],R=function(t){Object(l["a"])(a,t);var e=Object(c["a"])(a);function a(){return Object(s["a"])(this,a),e.apply(this,arguments)}return a}(m["c"]);R=Object(u["a"])([Object(m["a"])({name:"Warning"})],R);var $=R,S=$,I=Object(C["a"])(S,U,L,!1,null,null,null),T=I.exports,A=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("el-dropdown",{attrs:{"show-timeout":100,trigger:"click"}},[a("el-button",{attrs:{plain:""}},[t._v(" "+t._s(t.disableComment?"Comment: closed":"Comment: opened")+" "),a("i",{staticClass:"el-icon-caret-bottom el-icon--right"})]),a("el-dropdown-menu",{staticClass:"no-padding",attrs:{slot:"dropdown"},slot:"dropdown"},[a("el-dropdown-item",[a("el-radio-group",{staticStyle:{padding:"10px"},model:{value:t.disableComment,callback:function(e){t.disableComment=e},expression:"disableComment"}},[a("el-radio",{attrs:{label:!0}},[t._v(" Close comment ")]),a("el-radio",{attrs:{label:!1}},[t._v(" Open comment ")])],1)],1)],1)],1)},D=[],q=function(t){Object(l["a"])(a,t);var e=Object(c["a"])(a);function a(){return Object(s["a"])(this,a),e.apply(this,arguments)}return Object(i["a"])(a,[{key:"disableComment",get:function(){return this.value},set:function(t){this.$emit("input",t)}}]),a}(m["c"]);Object(u["a"])([Object(m["b"])({required:!0})],q.prototype,"value",void 0),q=Object(u["a"])([Object(m["a"])({name:"CommentDropdown"})],q);var E=q,P=E,V=Object(C["a"])(P,A,D,!1,null,null,null),N=V.exports,z=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("el-dropdown",{attrs:{"hide-on-click":!1,"show-timeout":100,trigger:"click"}},[a("el-button",{attrs:{plain:""}},[t._v(" Platfroms("+t._s(t.platforms.length)+") "),a("i",{staticClass:"el-icon-caret-bottom el-icon--right"})]),a("el-dropdown-menu",{attrs:{slot:"dropdown"},slot:"dropdown"},[a("el-checkbox-group",{staticStyle:{padding:"5px 15px"},model:{value:t.platforms,callback:function(e){t.platforms=e},expression:"platforms"}},t._l(t.platformsOptions,(function(e){return a("el-checkbox",{key:e.key,attrs:{label:e.key}},[t._v(" "+t._s(e.name)+" ")])})),1)],1)],1)},B=[],M=function(t){Object(l["a"])(a,t);var e=Object(c["a"])(a);function a(){var t;return Object(s["a"])(this,a),t=e.apply(this,arguments),t.platformsOptions=[{key:"a-platform",name:"a-platform"},{key:"b-platform",name:"b-platform"},{key:"c-platform",name:"c-platform"}],t}return Object(i["a"])(a,[{key:"platforms",get:function(){return this.value},set:function(t){this.$emit("input",t)}}]),a}(m["c"]);Object(u["a"])([Object(m["b"])({required:!0})],M.prototype,"value",void 0),M=Object(u["a"])([Object(m["a"])({name:"PlatformDropdown"})],M);var H=M,J=H,W=Object(C["a"])(J,z,B,!1,null,null,null),G=W.exports,K=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("el-dropdown",{attrs:{"show-timeout":100,trigger:"click"}},[a("el-button",{attrs:{plain:""}},[t._v(" Link "),a("i",{staticClass:"el-icon-caret-bottom el-icon--right"})]),a("el-dropdown-menu",{staticClass:"no-padding",staticStyle:{width:"400px"},attrs:{slot:"dropdown"},slot:"dropdown"},[a("el-form-item",{staticStyle:{"margin-bottom":"0px"},attrs:{"label-width":"0px",prop:"sourceURL"}},[a("el-input",{attrs:{placeholder:"Please enter the content"},model:{value:t.sourceURL,callback:function(e){t.sourceURL=e},expression:"sourceURL"}},[a("template",{slot:"prepend"},[t._v(" URL ")])],2)],1)],1)],1)},Q=[],X=function(t){Object(l["a"])(a,t);var e=Object(c["a"])(a);function a(){return Object(s["a"])(this,a),e.apply(this,arguments)}return Object(i["a"])(a,[{key:"sourceURL",get:function(){return this.value},set:function(t){this.$emit("input",t)}}]),a}(m["c"]);Object(u["a"])([Object(m["b"])({required:!0})],X.prototype,"value",void 0),X=Object(u["a"])([Object(m["a"])({name:"SourceUrlDropdown"})],X);var Y=X,Z=Y,tt=Object(C["a"])(Z,K,Q,!1,null,null,null),et=tt.exports,at=function(t){Object(l["a"])(a,t);var e=Object(c["a"])(a);function a(){var t;return Object(s["a"])(this,a),t=e.apply(this,arguments),t.validateRequire=function(e,a,r){""===a?("imageURL"===e.field?t.$message({message:"Upload cover image is required",type:"error"}):t.$message({message:e.field+" is required",type:"error"}),r(new Error(e.field+" is required"))):r()},t.validateSourceUrl=function(e,a,r){a?Object(p["c"])(a)?r():(t.$message({message:"Invalid URL",type:"error"}),r(new Error("Invalid URL"))):r()},t.postForm=Object.assign({},d["b"]),t.loading=!1,t.userListOptions=[],t.rules={imageURL:[{validator:t.validateRequire}],title:[{validator:t.validateRequire}],fullContent:[{validator:t.validateRequire}],sourceURL:[{validator:t.validateSourceUrl,trigger:"blur"}]},t.tinymceActive=!0,t}return Object(i["a"])(a,[{key:"created",value:function(){if(this.isEdit){var t=this.$route.params&&this.$route.params.id;this.fetchData(parseInt(t))}this.tempTagView=Object.assign({},this.$route)}},{key:"deactivated",value:function(){this.tinymceActive=!1}},{key:"activated",value:function(){this.tinymceActive=!0}},{key:"fetchData",value:function(){var t=Object(n["a"])(regeneratorRuntime.mark((function t(e){var a,r,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.prev=0,t.next=3,Object(d["c"])(e,{});case 3:a=t.sent,r=a.data,this.postForm=r.article,this.postForm.title+="   Article Id:".concat(this.postForm.id),this.postForm.abstractContent+="   Article Id:".concat(this.postForm.id),o="zh"===this.lang?"编辑文章":"Edit Article",this.setTagsViewTitle(o),this.setPageTitle(o),t.next=16;break;case 13:t.prev=13,t.t0=t["catch"](0),console.error(t.t0);case 16:case"end":return t.stop()}}),t,this,[[0,13]])})));function e(e){return t.apply(this,arguments)}return e}()},{key:"setTagsViewTitle",value:function(t){var e=this.tempTagView;e&&(e.title="".concat(t,"-").concat(this.postForm.id),v["a"].updateVisitedView(e))}},{key:"setPageTitle",value:function(t){document.title="".concat(t," - ").concat(this.postForm.id)}},{key:"submitForm",value:function(){var t=this;this.$refs.postForm.validate((function(e){if(!e)return console.error("Submit Error!"),!1;t.loading=!0,t.$notify({title:"Success",message:"The post published successfully",type:"success",duration:2e3}),t.postForm.status="published",setTimeout((function(){t.loading=!1}),500)}))}},{key:"draftForm",value:function(){0!==this.postForm.fullContent.length&&0!==this.postForm.title.length?(this.$message({message:"The draft saved successfully",type:"success",showClose:!0,duration:1e3}),this.postForm.status="draft"):this.$message({message:"Title and detail content are required",type:"warning"})}},{key:"getRemoteUserList",value:function(){var t=Object(n["a"])(regeneratorRuntime.mark((function t(e){var a,r;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,Object(f["b"])({name:e});case 2:if(a=t.sent,r=a.data,r.items){t.next=6;break}return t.abrupt("return");case 6:this.userListOptions=r.items.map((function(t){return t.name}));case 7:case"end":return t.stop()}}),t,this)})));function e(e){return t.apply(this,arguments)}return e}()},{key:"abstractContentLength",get:function(){return this.postForm.abstractContent.length}},{key:"lang",get:function(){return b["a"].language}},{key:"timestamp",get:function(){return+new Date(this.postForm.timestamp)},set:function(t){this.postForm.timestamp=+new Date(t)}}]),a}(m["c"]);Object(u["a"])([Object(m["b"])({default:!1})],at.prototype,"isEdit",void 0),at=Object(u["a"])([Object(m["a"])({name:"ArticleDetail",components:{CommentDropdown:N,PlatformDropdown:G,SourceUrlDropdown:et,MaterialInput:h["a"],Sticky:g["a"],Tinymce:w["a"],UploadImage:_,Warning:T}})],at);var rt=at,ot=rt,nt=(a("7162"),a("fa48"),Object(C["a"])(ot,r,o,!1,null,"2a232006",null));e["a"]=nt.exports},2305:function(t,e,a){},7162:function(t,e,a){"use strict";a("cc36")},"9d25":function(t,e,a){"use strict";a.d(e,"b",(function(){return o})),a.d(e,"d",(function(){return n})),a.d(e,"c",(function(){return s})),a.d(e,"a",(function(){return i})),a.d(e,"f",(function(){return l})),a.d(e,"e",(function(){return c}));var r=a("b32d"),o={id:0,status:"draft",title:"",fullContent:"",abstractContent:"",sourceURL:"",imageURL:"",timestamp:"",platforms:["a-platform"],disableComment:!1,importance:0,author:"",reviewer:"",type:"",pageviews:0},n=function(t){return Object(r["a"])({url:"/articles",method:"get",params:t})},s=function(t,e){return Object(r["a"])({url:"/articles/".concat(t),method:"get",params:e})},i=function(t){return Object(r["a"])({url:"/articles",method:"post",data:t})},l=function(t,e){return Object(r["a"])({url:"/articles/".concat(t),method:"put",data:e})},c=function(t){return Object(r["a"])({url:"/pageviews",method:"get",params:t})}},c4fc:function(t,e,a){"use strict";a("2305")},c7d3:function(t,e,a){},cc36:function(t,e,a){t.exports={menuBg:"#304156",menuText:"#bfcbd9",menuActiveText:"#409eff"}},fa48:function(t,e,a){"use strict";a("c7d3")}}]);