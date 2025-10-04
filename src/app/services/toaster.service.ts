import {Injectable} from '@angular/core';
import { MessageService } from 'primeng/api';

@Injectable({
  providedIn: 'root'
})
export class ToasterService {
  constructor(private messageService:MessageService) {
  }

  /**
   * Shows a success toast message
   * @param message The message to be displayed in the toast
   * @param title The title of the toast
   */
  SuccessToster(message: string, title?: string): void {
    this.messageService.add({ severity: 'success',  summary: title || 'Success', detail: `${message}` });
  }

  infoToster(message: string, title?: string): void {
    this.messageService.add({ severity: 'info',  summary: title||'Info', detail: `${message}` });
  }

  warningToster(message: string, title?: string): void {
    this.messageService.add({ severity: 'warn',  summary: title||'Warning', detail: `${message}` });
  }

  errorToster(message: string, title?: string): void {
    this.messageService.add({ severity: 'error', summary:title||'Error', detail: `${message}` });
  }
}
