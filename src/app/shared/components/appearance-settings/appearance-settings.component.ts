import { FormsModule } from '@angular/forms';
import { Component, inject } from '@angular/core';
import { DialogModule } from "primeng/dialog";
import { ToggleSwitchModule } from "primeng/toggleswitch";
import { ButtonModule } from "primeng/button";
import { ToasterService } from '../../../services/toaster.service';
import { ThemeService } from '../../../services/theme.service';
import { ColorPickerModule } from "primeng/colorpicker";

@Component({
  selector: 'app-appearance-settings',
  imports: [DialogModule, ToggleSwitchModule, ButtonModule, ColorPickerModule,FormsModule],
  templateUrl: './appearance-settings.component.html',
  styleUrl: './appearance-settings.component.css'
})
export class AppearanceSettingsComponent {
  private theme = inject(ThemeService);
  private toast = inject(ToasterService);

  checkedDarkMode = this.theme.getDarkMode();
  color: string = this.theme.getAccent();
displaySettingsDialog: boolean = false;
showSettingsDialog() {
    this.displaySettingsDialog = true;
  }
  saveAppearance() {
    this.theme.saveAndApplyAccent(this.color);
    this.theme.setDarkMode(this.checkedDarkMode);
    this.toast.SuccessToster('Appearance has been changed successfully !');
    this.displaySettingsDialog = false;
  }
}
