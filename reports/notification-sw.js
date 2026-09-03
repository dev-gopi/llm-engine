self.addEventListener('notificationclick',event=>{
  event.notification.close();
  event.waitUntil(clients.matchAll({type:'window',includeUncontrolled:true}).then(windows=>{
    const existing=windows[0];
    return existing?existing.focus():clients.openWindow('training_report.html');
  }));
});
