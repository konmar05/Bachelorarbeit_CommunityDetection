export const writeThesis = (literature, template) => {
    var thesis;
    while(!literature.isUnderstood){
        literature.openAgain();
    }
    template.fill();
    thesis = template.toPDF();
    return thesis;
}
